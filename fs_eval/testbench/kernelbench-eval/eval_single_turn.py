from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import os
from typing import List, Union
import numpy as np
import ray
from reward.cuda_rm import compute_score
from dataclasses import asdict
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'
import pandas as pd
import re
import tqdm
from reward.refine import Refine
from loguru import logger
from llmkit_data.utils.json import read_jsonl, write_jsonl

def _extract_cuda_code(text: str) -> str:
    # 当有多个代码块时，提取最后一个代码块
    codeblock_seps = ["python"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```({languages_pattern})"
    pattern = re.compile(codeblock_start + r"\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return code_content
    return None


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([
        estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)
    ])


def compute_score_remote(solution_str: str, ground_truth: str, config) -> Refine:
    """Ray remote function for computing score with GPU"""
    return compute_score(solution_str, ground_truth, config)


def compute_pass_k(dataset, config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "0",
                    "BPEX_NO_WARN_ON_UNTUNED_CASE": "1",
                    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                }
            },
            num_cpus=config.ray_init.num_cpus,
            num_gpus=4,
        )
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    logger.info(f"Cluster resources: {cluster_resources}")
    logger.info(f"Available resources: {available_resources}")

    futures = []
    task_mapping = []
    n_sample = config.data.n_samples

    rm_req_executor = ThreadPoolExecutor(max_workers=128)

    for idx, row in dataset.iterrows():
        ground_truth = row["reward_model"]["ground_truth"]
        for solution_str in row["responses"]:
            futures.append(
                rm_req_executor.submit(
                    compute_score_remote, solution_str, ground_truth, config
                )
            )
            task_mapping.append((idx, solution_str))

    # Organize results
    scores = [[] for _ in range(len(dataset))]
    for (idx, solution_str), score in tqdm.tqdm(
        zip(task_mapping, as_completed(futures)),
        desc="Compute scores",
        total=len(futures),
    ):
        score = score.result()
        print(score)
        scores[idx].append(asdict(score))

    if n_sample >= 5:
        k = [1, 3, 5]
    elif n_sample >= 10:
        k = [1, 3, 5, 10]
    else:
        k = [1]

    ## pass_at_k
    pass_at_k = {}
    for top_k in k:
        pass_at_k[top_k] = estimate_pass_at_k(
            num_samples=[len(scores[prompt_idx]) for prompt_idx in range(len(scores))],
            num_correct=[
                sum(1 for score in scores[prompt_idx] if score["passed"])
                for prompt_idx in range(len(scores))
            ],
            k=top_k,
        ).tolist()
        logger.info(f"Pass@{top_k}: {np.mean(pass_at_k[top_k])}")

    ## compile_pass_at_k
    compile_pass_at_k = {}
    for top_k in k:
        compile_pass_at_k[top_k] = estimate_pass_at_k(
            num_samples=[len(scores[prompt_idx]) for prompt_idx in range(len(scores))],
            num_correct=[
                sum(1 for score in scores[prompt_idx] if score["compiled"])
                for prompt_idx in range(len(scores))
            ],
            k=top_k,
        ).tolist()
        logger.info(f"Compile Pass@{top_k}: {np.mean(compile_pass_at_k[top_k])}")

    # dataset["pass_k"] = pass_at_k
    # Add scores to dataset
    dataset["result"] = scores
    return pass_at_k

def main(sample_file_path):
    samples = read_jsonl(sample_file_path)
    samples_id = {}
    for sample in samples:
        id = sample[""]

if __name__ == "__main__":
    main()
