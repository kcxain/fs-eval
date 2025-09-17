from typing import List
from fs_eval.uinfer.engine import VLLMInferenceEngine, SamplingParams, UinferOutput
from fs_eval.evaluator import eval
from fs_eval.datasets_map import dataset_input_map, support_dataset
from fs_eval.utils.io import read_jsonl, write_jsonl
from fs_eval.utils.regex import extract_code_block
from loguru import logger
from pathlib import Path
import os

gpu_groups = [[0, 1], [2, 3]]

OUTPUTS = Path("./outputs/eval/")


def read_data(path):
    data = read_jsonl(path)
    item = data[0]
    assert "question" in item, "sample data must contain 'question' field"
    assert "task_id" in item, "sample data must contain 'task_id' field"
    return data


class CodeSampler:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        params: SamplingParams,
        gpu_groups: List[List[str]] = gpu_groups,
    ):
        self.model_name = model_name
        self.params = params
        self.engine = VLLMInferenceEngine(
            pretrained_model=model_path, gpu_groups=gpu_groups, sampling_params=params
        )
        Warning("Call `close()` to release the GPU memory after sampling.")
        logger.add(f"logs/eval_{self.model_name}.log", rotation="10 MB")
        logger.info(f"Model: {model_name}, Path: {model_path}")
        logger.info(f"Sampling params: {params}")

    def sample(
        self,
        data,
        sample_n: int,
        output_path: str,
    ):
        prompts = []
        tokenizer = self.engine.get_tokenizer()
        for d in data:
            chat_list = d["question"]
            prompt = tokenizer.apply_chat_template(
                chat_list, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        outputs: List[UinferOutput] = self.engine.run(
            prompts=prompts,
            sample_num=sample_n,
            parse_output=extract_code_block,
        )
        save_data = []
        for i, d in enumerate(data):
            task_id = d["task_id"]
            for out, eout in zip(outputs[i].outputs, outputs[i].extracted_outputs):
                save_data.append({
                    "task_id": task_id,
                    "completion": eout,
                    "raw_completion": out,
                })
        write_jsonl(save_data, output_path)

    def evaluate(self, input_path: str, dataset: str, output_path: str = None):
        """
        input_path: path to the sampled jsonl file
        output_path: path to save the evaluation result
        dataset: dataset name, e.g., "humaneval", "verilog_eval"
        """
        logger.add(f"logs/eval_{dataset}_{self.model_name}.log", rotation="10 MB")
        data = read_jsonl(input_path)
        if not output_path:
            os.makedirs(OUTPUTS / dataset, exist_ok=True)
        temperature = self.params.temperature
        top_p = self.params.top_p
        top_k = self.params.top_k
        output_path = (
            OUTPUTS
            / dataset
            / f"{self.model_name}_temp{temperature}_p{top_p}_k{top_k}.jsonl"
        )
        eval(dataset, data, output_path)

    def evaluate_model(
        self,
        data,
        sample_n,
        dataset: str,
        output_path: str = None,
        prompt_type: str = None,
    ):
        """
        data: list of dict, each dict contains "task_id" and "question" fields
        sample_n: number of samples to generate for each question
        dataset: dataset name, e.g., "humaneval", "verilog_eval"
        output_path: path to save the evaluation result
        """
        logger.info(f"Evaluating model {self.model_name} on dataset {dataset}")
        if not output_path:
            os.makedirs(OUTPUTS / dataset, exist_ok=True)
        if (data is None) or (prompt_type is not None):
            from fs_eval.prompt_builder import build_prompt

            data = build_prompt(dataset, prompt_type)

        if "completion" in data[0]:
            logger.warning(
                "The input data already contains 'completion' field, "
                "we use the existing 'completion'."
            )
            eval(dataset, data, output_path)
        else:
            temperature = self.params.temperature
            top_p = self.params.top_p
            top_k = self.params.top_k
            output_path = (
                OUTPUTS
                / dataset
                / f"{self.model_name}_temp{temperature}_p{top_p}_k{top_k}_sample{sample_n}.jsonl"
            )
            sample_file_path = output_path.with_suffix(".sample.jsonl")
            self.sample(data, sample_n, sample_file_path)
            eval(dataset, read_jsonl(sample_file_path), output_path)

    def close(self):
        self.engine.stop_workers()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/personal/S/wangshuohit/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-7B")
    parser.add_argument(
        "--dataset", nargs="*", help="dataset name", default="kernelbench"
    )
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--sample_n", type=int, default=5)
    parser.add_argument("--prompt_type", type=str, default=None)

    args = parser.parse_args()
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    sampler = CodeSampler(
        model_path=args.model_path,
        model_name=args.model_name,
        params=params,
    )
    if args.dataset == ["all"]:
        args.dataset = support_dataset
    for dataset in args.dataset:
        if args.data_path is not None:
            assert len(args.dataset) == 1, (
                "--data_path is provided, please only specify one dataset"
            )
            data_path = args.data_path
            data = read_data(data_path)
        else:
            data = None
        sampler.evaluate_model(
            data=data,
            sample_n=args.sample_n,
            dataset=dataset,
            output_path=args.output_path,
            prompt_type=args.prompt_type,
        )
    sampler.close()
