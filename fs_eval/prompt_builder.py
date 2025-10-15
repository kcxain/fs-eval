import json

from fs_eval.datasets_map import dataset_problem_map
from fs_eval.utils.io import read_jsonl, write_jsonl
from loguru import logger

PROMPT_BUILDERS = {}


def register_prompt_builder(name):
    def wrapper(cls):
        PROMPT_BUILDERS[name] = cls()
        return cls

    return wrapper


class BasePromptBuilder:
    def __init__(self, prompt_type="direct"):
        assert prompt_type in ["direct", "cot"], "prompt_type must be 'direct' or 'cot'"
        self.prompt_type = prompt_type

    def load_problem_data(self, dataset_name):
        file_path = dataset_problem_map[dataset_name]
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def build_prompt(self, dataset_name, problem):
        raise NotImplementedError


@register_prompt_builder("compute-eval")
class ComputeEvalPromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        match self.prompt_type:
            case "direct":
                return f"Write a CUDA function to perform the following task:\n{problem['description']}\n"
            case "cot":
                from fs_eval.prompts.compute_eval import (
                    system_prompt,
                    user_prompt,
                    extract_header_files_from_problem,
                    HEADER_FILES_PROMPT_TEMPLATE,
                )

                header = extract_header_files_from_problem(problem)
                header_prompt = HEADER_FILES_PROMPT_TEMPLATE.format(header_files=header)
                return {
                    "task_id": problem["task_id"],
                    "question": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt.replace(
                                "__CUDA_PROBLEM__", problem["prompt"]
                            )
                            + "\n"
                            + header_prompt,
                        },
                    ],
                    "data_source": dataset_name,
                }


@register_prompt_builder("humaneval")
class HumanEvalPromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        if self.prompt_type == "direct":
            return f"Please complete the following function:\n{problem['prompt']}\n"
        else:
            from fs_eval.prompts.humaneval import system_prompt, user_prompt

            return {
                "task_id": problem["task_id"],
                "question": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            problem="Implete the following function based on its signature. Your final answer should include the whole function implementation.",
                            function_signature=problem["prompt"].strip(),
                        ),
                    },
                ],
                "data_source": dataset_name,
            }


@register_prompt_builder("leetcode")
class LeetCodePromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        if self.prompt_type == "direct":
            return f"Implement a solution for this LeetCode problem:\n{problem['description']}\n"
        else:
            from fs_eval.prompts.humaneval import system_prompt, user_prompt

            return {
                "task_id": problem["task_id"],
                "question": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            problem=problem["problem_description"],
                            function_signature=problem["starter_code"],
                        ),
                    },
                ],
                "data_source": dataset_name,
            }


@register_prompt_builder("verilog_eval_v1-human")
class VerilogEvalHPromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        if self.prompt_type == "direct":
            return f"Write a Verilog module to satisfy the requirement:\n{problem['prompt']}\n"
        else:
            from fs_eval.prompts.verilog_eval import system_prompt, user_prompt

            return {
                "task_id": problem["task_id"],
                "question": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            question=problem["detail_description"],
                            module_head=problem["prompt"],
                        ),
                    },
                ],
                "data_source": dataset_name,
            }


@register_prompt_builder("verilog_eval_v1-machine")
class VerilogEvalMPromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        if self.prompt_type == "direct":
            return f"Generate a Verilog implementation for:\n{problem['prompt']}\n"
        else:
            from fs_eval.prompts.verilog_eval import system_prompt, user_prompt

            return {
                "task_id": problem["task_id"],
                "question": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            question=problem["detail_description"],
                            module_head=problem["prompt"],
                        ),
                    },
                ],
                "data_source": dataset_name,
            }


@register_prompt_builder("kernelbench")
class KernelBenchPromptBuilder(BasePromptBuilder):
    def build_prompt(self, dataset_name, problem):
        dataset_id = dataset_name
        match self.prompt_type:
            case "direct":
                return f"Write a CUDA kernel to perform the following task:\n{problem['description']}\n"
            case "cot":
                from fs_eval.prompts.kernelbench import user_prompt, system_prompt

                pytorch_module = problem["ground_truth"]["pytorch_module"]
                level_id = str(problem["level"])
                op_name = problem["extra_info"]["op_name"]
                problem_id = problem["extra_info"]["problem_id"]
                output = {
                    "task_id": level_id + "-" + str(problem_id),
                    "question": [
                        # {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt.replace(
                                "__PYTORCH_MODULE__", pytorch_module.strip()
                            ),
                        },
                    ],
                    "data_source": f"{dataset_id}/level_{level_id}",
                    "ability": "code",
                    "answer": "",
                    "raw_problem": pytorch_module,
                    "level": level_id,
                    "type": "",
                    "ground_truth": {
                        "pytorch_module": pytorch_module,
                    },
                    "style": "cuda-sandbox-v2",
                    "extra_info": {"op_name": op_name, "problem_id": problem_id},
                }
                return output

            case _:
                raise ValueError(f"Unsupported prompt type: {self.prompt_type}")


def build_prompt(dataset_name, prompt_type="direct"):
    if dataset_name not in PROMPT_BUILDERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    problem_data = read_jsonl(dataset_problem_map[dataset_name])
    builder = PROMPT_BUILDERS[dataset_name]
    builder.prompt_type = prompt_type
    prompts = []
    for problem in problem_data:
        prompt = builder.build_prompt(dataset_name, problem)
        if isinstance(prompt, dict):
            prompts.append(prompt)
        else:
            prompts.append({
                "task_id": problem["task_id"],
                "question": prompt,
                "data_source": dataset_name,
            })
    write_jsonl(prompts, f"fs_eval/data/{prompt_type}/{dataset_name}.jsonl")
    logger.info(
        f"Built {len(prompts)} prompts for dataset {dataset_name} with prompt type {prompt_type}"
    )
    logger.info(f"Saved prompts to fs_eval/data/{prompt_type}/{dataset_name}.jsonl")
    logger.info(f"Example prompt: {prompts[0]}")
    return prompts
