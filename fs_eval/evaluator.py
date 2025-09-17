import json
from fs_eval.utils.io import write_jsonl
from loguru import logger
from fs_eval.datasets_map import dataset_problem_map

EVALUATORS = {}


def register_evaluator(name):
    def wrapper(cls):
        EVALUATORS[name] = cls()
        return cls

    return wrapper


class BaseEvaluator:
    def load_problem_data(self, dataset_name):
        file_path = dataset_problem_map[dataset_name]
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def get_problem_path(self, dataset_name):
        return dataset_problem_map[dataset_name]

    def evaluate(self, dataset_name, sample_data, output_path):
        raise NotImplementedError


@register_evaluator("compute-eval")
class ComputeEvalEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from compute_eval.evaluation import evaluate_functional_correctness_from_data

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


@register_evaluator("humaneval")
class HumanEvalEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from human_eval.evaluation import evaluate_functional_correctness_from_data

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


@register_evaluator("leetcode")
class LeetCodeEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from eval_lcd.evaluate import (
            evaluate_functional_correctness_from_data,
        )

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


@register_evaluator("verilog_eval_v1-human")
class VerilogEvalVEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from verilog_eval.evaluation import evaluate_functional_correctness_from_data

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


@register_evaluator("verilog_eval_v1-machine")
class VerilogEvalMEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from verilog_eval.evaluation import evaluate_functional_correctness_from_data

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


@register_evaluator("kernelbench")
class KernelBenchEvaluator(BaseEvaluator):
    def evaluate(self, dataset_name, sample_data, output_path):
        from kernelbench_eval.evaluation import (
            evaluate_functional_correctness_from_data,
        )

        logger.info(f"Start Evalueting {dataset_name}...")
        problem = self.load_problem_data(dataset_name)
        results = evaluate_functional_correctness_from_data(sample_data, problem)
        write_jsonl(results, output_path)


def eval(dataset_name, sample_data, output_path="./tmp.jsonl"):
    if dataset_name not in EVALUATORS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    assert "completion" in sample_data[0], "input_data must contain 'completion' field"
    assert "task_id" in sample_data[0], "input_data must contain 'problem_id' field"
    return EVALUATORS[dataset_name].evaluate(dataset_name, sample_data, output_path)
