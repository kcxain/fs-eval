import time
import re
import warnings
import contextlib
import json
from datetime import datetime
from multiprocessing import Process
from collections import Counter

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import os

import ray
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.fs import copy_local_path_from_hdfs
import torch
import wandb
import pandas as pd
import numpy as np

MegavisionMetricsCtx = None
import response_post_proc
import cuda_rm
from ray_trainer import RayPPOTrainer

import ray
import hydra
import omegaconf
from omegaconf import DictConfig


def _select_rm_score_fn(reward_style):
    if "cuda-sandbox" in reward_style:
        return cuda_rm.compute_score
    else:
        raise NotImplementedError


def post_process_solution_str(config, solution_str, eos_token):
    solution_str = solution_str.rsplit(eos_token, 1)[0]
    if config.reward_model.use_last_response == "summarize":
        solution_str_post_proc = response_post_proc.summary_postprocess(
            solution_str,
            last_response_sep=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict,
        )
    elif config.reward_model.use_last_response == "lastcodeblock":
        solution_str_post_proc = response_post_proc.last_codeblock_postprocess(
            solution_str,
            codeblock_seps=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict,
        )
    else:
        solution_str_post_proc = solution_str
    return solution_str_post_proc


@ray.remote(num_cpus=1)
class RemoteClient:
    """
    A centralized remote client that pipelines any function with generation at [EOS]
    """

    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.results = {}

        self.call_cuda_rm = ray.remote(num_cpus=1)(cuda_rm.compute_score)

    def clear(self):
        self.results = {}

    def get_num_pending_outputs(self):
        """Return the number of outputs, whose result is not claimed"""
        return len(self.results)

    async def add_requests(self, req_id, input_ids, ground_truth):
        solution_str = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        solution_str_post_proc = post_process_solution_str(
            self.config, solution_str, eos_token=self.tokenizer.eos_token
        )
        result_future = self.call_cuda_rm.remote(
            solution_str_post_proc, ground_truth, config=self.config
        )

        assert req_id not in self.results, f"{req_id} already exists"
        self.results[req_id] = result_future

    async def get_results(self, req_id):
        if req_id not in self.results:
            return None

        assert req_id in self.results, f"{req_id} not found"
        result_future = self.results.pop(req_id)
        return await result_future


try:
    from nltk.util import ngrams
except ImportError:
    ngrams = None
    warnings.warn("nltk not installed, please install nltk. Disable diversity metrics.")

import math


class RewardManager:
    def __init__(self, tokenizer, config, logger: Tracking, rm_name="train") -> None:
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_table = []
        self.rm_name = rm_name
        self.config = config
        self.rm_req_executor = ThreadPoolExecutor(
            max_workers=int(self.config.reward_model.get("reward_executor_maxnum", 128))
        )
        self.mean = self.config.reward_model.mean
        self.std = self.config.reward_model.std
        self.need_punish_duplicate = self.config.reward_model.get(
            "need_punish_duplicate", False
        )
        self.punish_score = self.config.reward_model.get(
            "punish_score", "rule-lighteval/MATH_v2:-1,code-sandbox:0"
        )
        self.punish_score = dict(
            map(
                lambda x: (x.split(":")[0], float(x.split(":")[1])),
                self.punish_score.split(","),
            )
        )
        self.need_punish_trunc = self.config.reward_model.get(
            "need_punish_trunc", False
        )
        self.trunc_punish_score = self.config.reward_model.get("trunc_punish_score", -5)
        self.len_ema_without_overlong = self.config.reward_model.get(
            "len_ema_without_overlong", False
        )
        self.length_ema_method = self.config.reward_model.get(
            "length_ema_method", "mean"
        )
        assert (
            not self.len_ema_without_overlong
            or self.need_punish_trunc
            or self.config.algorithm.mask_overlong
            or self.config.algorithm.overlong_punish != "v0"
        ), (
            "len_ema_without_overlong is True, so self.need_punish_trunc or mask_overlong must be true."
        )
        self.len_ema_lambda = self.config.reward_model.get("len_ema_lambda", 1)
        self.len_ema = {}
        self.len_ema_json = self.config.reward_model.get("len_ema_json", None)

    def update_len_ema(self, data: DataProto):
        index = data.non_tensor_batch["index"]
        lengths = data.batch["attention_mask"][
            :, self.config.data.max_prompt_length :
        ].sum(-1)

        len_lst = {}
        for idx, length in enumerate(lengths):
            if index[idx] not in len_lst:
                len_lst[index[idx]] = []
            if self.len_ema_without_overlong is True:
                if "max_new_tokens" in data.non_tensor_batch:
                    if length >= data.non_tensor_batch["max_new_tokens"][idx]:
                        continue
                if length >= self.config.data.max_response_length:
                    continue
            len_lst[index[idx]].append(length)

        for idx in len_lst:
            lst = len_lst[idx]
            if not lst:
                default_value = torch.tensor(self.config.data.max_response_length)
                cur_stat = self.len_ema.get(idx, default_value)
            else:
                if self.length_ema_method == "mean":
                    cur_stat = sum(lst) / len(lst)
                elif self.length_ema_method == "median":
                    cur_stat = sorted(lst)[len(lst) // 2]
                else:
                    raise ValueError(
                        f"Unknown length_ema_method: {self.length_ema_method}"
                    )
            if idx not in self.len_ema:
                self.len_ema[idx] = cur_stat
            else:
                self.len_ema[idx] = (1 - self.len_ema_lambda) * self.len_ema[
                    idx
                ] + self.len_ema_lambda * cur_stat

        mean_len_per_prompt = [self.len_ema[idx].item() for idx in index]
        return mean_len_per_prompt

    def __call__(
        self,
        data: DataProto,
        global_step=None,
        need_norm=True,
        is_validation=False,
        return_dict=False,
    ):
        """We will expand this function gradually based on the available datasets"""
        response_ids = data.batch["input_ids"][:, self.config.data.max_prompt_length :]

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        raw_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        format_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        len_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        idx_tensor = torch.zeros(
            response_ids.shape[0], dtype=torch.int64, device=response_ids.device
        )
        already_print_data_sources = {}
        save_to_hdfs = []
        rm_res_future_list = []
        if (
            global_step is not None
            and global_step % self.config.trainer.logger_step_interval == 0
        ):
            self.log_table = []

        mean_len_per_prompt = self.update_len_ema(data)
        current_mean_len = (
            data.batch["attention_mask"][:, self.config.data.max_prompt_length :]
            .sum(-1)
            .float()
            .mean()
            .item()
        )

        def get_rm_score(idx):
            data_item = data[idx]  # DataProtoItem

            prompt_ids = data_item.batch["input_ids"][
                : self.config.data.max_prompt_length
            ]
            response_ids = data_item.batch["input_ids"][
                self.config.data.max_prompt_length :
            ]

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = (
                data_item.batch["attention_mask"][:prompt_length].sum().item()
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_length = response_ids.shape[-1]
            valid_response_length = (
                data_item.batch["attention_mask"][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=False
            )
            solution_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )

            solution_str_post_proc = post_process_solution_str(
                config=self.config,
                solution_str=solution_str,
                eos_token=self.tokenizer.eos_token,
            )

            format_reward = 0
            pause_tokens_index = None
            thinking_len = 0
            # get prompt uuid
            data_uid = data_item.non_tensor_batch["uid"]

            # select rm_score
            reward_style = data_item.non_tensor_batch["reward_model"]["style"]
            compute_score_fn = _select_rm_score_fn(reward_style)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            score_fn_inputs = {
                "batch_info": data_item.batch,
                "tokenizer": self.tokenizer,
                "solution_str": solution_str_post_proc,
                "ground_truth": ground_truth,
                "config": self.config,
                "data_uid": data_uid,
                "solution_len": valid_response_length,
                "solution_ids": valid_response_ids,
                "rm_name": self.rm_name,
                "pause_tokens_index": pause_tokens_index,
            }

            score = compute_score_fn(**score_fn_inputs)
            is_trunc = (response_length == valid_response_length) and score == -1

            ngram = (
                list(ngrams(valid_response_ids.tolist(), 2))
                if ngrams is not None
                else []
            )

            return_dict = {
                "prompt_str": prompt_str,
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "reward_style": reward_style,
                "valid_response_length": valid_response_length,
                "score": score,
                "is_para_dup": False,
                "is_trunc": is_trunc,
                "idx": idx,
                "solution_str_post_proc": solution_str_post_proc,
                "ngram": ngram,
                "format_reward": format_reward,
                "pause_tokens_index": pause_tokens_index,
                "thinking_len": thinking_len,
                "global_index": data_item.non_tensor_batch["index"],
            }

            return return_dict

        for i in range(len(data)):
            rm_res_future_list.append(self.rm_req_executor.submit(get_rm_score, i))
        dup_cnt = 0
        dup_lens = []
        not_dup_lens = []
        from tqdm import tqdm

        all_final_scores = []
        all_extra_timing = defaultdict(list)

        log_table_interval = 1
        if self.config.trainer.num_cases_to_wandb > 0:
            log_table_interval = len(data) // self.config.trainer.num_cases_to_wandb
        all_final_scores_to_lens = defaultdict(list)
        i_to_idx = []

        for i, res in tqdm(
            enumerate(as_completed(rm_res_future_list)),
            total=len(data),
            desc="get_rm_score",
        ):
            output_dict = res.result()
            prompt_str = output_dict["prompt_str"]
            solution_str = output_dict["solution_str"]
            ground_truth = output_dict["ground_truth"]
            reward_style = output_dict["reward_style"]
            valid_response_length = output_dict["valid_response_length"]
            score = output_dict["score"]
            score_msg = ""
            if isinstance(score, dict):
                if "compile_time" in score:
                    all_extra_timing["compile_time"].append(score["compile_time"])
                if "execute_time" in score:
                    all_extra_timing["execute_time"].append(score["execute_time"])
                score_msg = score["msg"]
                score = score["score"]
            is_para_dup = output_dict["is_para_dup"]
            is_trunc = output_dict["is_trunc"]
            idx = output_dict["idx"]
            solution_str_post_proc = output_dict["solution_str_post_proc"]
            format_reward = output_dict["format_reward"]
            global_index = output_dict["global_index"]

            i_to_idx.append(idx)

            if need_norm:
                score = (score - self.mean) / self.std
            raw_scores[idx, valid_response_length - 1] = score
            format_scores[idx, valid_response_length - 1] = format_reward
            len_scores[idx, valid_response_length - 1] = score

            dup_punish_reward = 0
            if is_para_dup:
                dup_cnt += 1
                dup_lens.append(valid_response_length)
                if self.need_punish_duplicate and not is_validation:
                    dup_punish_reward = self.punish_score.get(reward_style, -1)
                score += dup_punish_reward
            else:
                not_dup_lens.append(valid_response_length)
            if self.need_punish_trunc and is_trunc and not is_validation:
                score = self.trunc_punish_score
            if format_reward != 0:
                score += format_reward
            reward_tensor[idx, valid_response_length - 1] = score
            idx_tensor[idx] = valid_response_length - 1

            assert is_divisible_by_0_point_1(score)
            all_final_scores.append(score)
            all_final_scores_to_lens[score].append(valid_response_length)

            if reward_style not in already_print_data_sources:
                already_print_data_sources[reward_style] = 0

            if (
                already_print_data_sources[reward_style]
                < self.config.trainer.num_cases_to_wandb
            ):
                already_print_data_sources[reward_style] += 1
                ground_truth = ""
                self.log_table.append(
                    [
                        global_index,
                        global_step,
                        prompt_str,
                        solution_str,
                        ground_truth,
                        score,
                        score_msg,
                        solution_str_post_proc,
                        is_para_dup,
                        is_trunc,
                        valid_response_length,
                    ]
                )
            save_to_hdfs.append(
                [
                    global_index,
                    idx,
                    global_step,
                    prompt_str,
                    solution_str,
                    ground_truth,
                    score,
                    score_msg,
                    solution_str_post_proc,
                    is_para_dup,
                    is_trunc,
                    valid_response_length,
                ]
            )

        final_counter = Counter(all_final_scores)

        all_final_scores_to_lens = {
            key: sum(value) / len(value)
            for key, value in all_final_scores_to_lens.items()
        }
        prefix = "" if not is_validation else "val/"
        log_data = {
            prefix + "dup/para_dup": dup_cnt / len(data),
            prefix + "dup/dup_response_len": sum(dup_lens) / max(1, len(dup_lens)),
            prefix + "dup/not_dup_response_len": sum(not_dup_lens)
            / max(1, len(not_dup_lens)),
            prefix + "current_mean_len": current_mean_len,
        }

        log_counter = {
            prefix + f"score_counter/final_{key}": value
            for key, value in final_counter.items()
        }
        # add score_counter by data_source
        if "data_source" in data.non_tensor_batch:
            score_by_data_source = defaultdict(list)
            for i, score in enumerate(all_final_scores):
                idx = i_to_idx[i]
                data_source = data.non_tensor_batch["data_source"][idx]
                score_by_data_source[data_source].append(score)
            for data_source, score_list in score_by_data_source.items():
                counter = Counter(score_list)
                log_counter.update(
                    {
                        f"{prefix}score_counter_by_source/{data_source.replace('/', '_')}_{key}": val
                        for key, val in counter.items()
                    }
                )

        log_score_to_lens = {
            prefix + f"score_to_lens/{key}": value
            for key, value in all_final_scores_to_lens.items()
        }
        log_score = {
            prefix + f"score/final": sum(all_final_scores)
            / max(1, len(all_final_scores)),
        }

        log_extra_metrics = {}
        for key, val in all_extra_timing.items():
            log_extra_metrics[f"timing/{key}_mean"] = np.mean(val)
            log_extra_metrics[f"timing/{key}_p99"] = np.percentile(val, 99)
        log_data = {
            **log_data,
            **log_counter,
            **log_score_to_lens,
            **log_score,
            **log_extra_metrics,
        }
        self.logger.log(data=log_data, step=global_step)

        log_table = None

        if self.config.trainer.num_cases_to_wandb > 0:
            log_table = {
                f"gen&score_{self.rm_name}_{global_step}": wandb.Table(
                    columns=[
                        "Index",
                        "Step",
                        "Prompt",
                        "Gen Sequence",
                        "GroundTruth",
                        "Score",
                        "ScoreMsg",
                        "Gen Sequence PostProc",
                        "Is_Dup",
                        "Is_Trunc",
                        "Len",
                    ],
                    data=self.log_table,
                )
            }
            if (
                not is_validation
                and global_step % self.config.trainer.logger_step_interval == 0
            ) or global_step == 1:
                # logger_step = global_step - global_step % self.config.trainer.logger_step_interval
                self.logger.log(log_table, step=global_step, backend="wandb")

        if return_dict:
            if not is_validation:
                return {
                    "reward_tensor": reward_tensor,
                    "raw_scores": raw_scores,
                    "len_scores": len_scores,
                    "idx_tensor": idx_tensor,
                }
            else:
                return {
                    "reward_tensor": reward_tensor,
                    "log_table": log_table,
                }

        if not is_validation:
            return reward_tensor, raw_scores, len_scores, idx_tensor
        else:
            return reward_tensor, log_table
