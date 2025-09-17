TEST_BENCH_PATH = "fs_eval/testbench/"

dataset_problem_map = {
    "compute-eval": TEST_BENCH_PATH
    + "compute-eval/data/combined_problems_033125.jsonl",
    # TODO: support spec level here
    "kernelbench": TEST_BENCH_PATH + "kernelbench-eval/data/KernelBench-level_1.jsonl",
    "humaneval": TEST_BENCH_PATH + "human-eval/data/HumanEval.jsonl",
    "leetcode": TEST_BENCH_PATH
    + "LeetCodeDataset/data/LeetCodeDataset-v0.3.1-test.jsonl",
    "verilog_eval_v1-human": TEST_BENCH_PATH
    + "VerilogEval_v1.0.0/data/VerilogEval_Human_Merged.jsonl",
    "verilog_eval_v1-machine": TEST_BENCH_PATH
    + "VerilogEval_v1.0.0/data/VerilogEval_Machine_Merged.jsonl",
}

dataset_input_map = {
    "compute-eval": "fs_eval/data/direct/compute_eval.jsonl",
    "kernelbench": "fs_eval/data/direct/KernelBench-level_1.jsonl",
    "humaneval": "fs_eval/data/direct/humaneval.jsonl",
    "leetcode": "fs_eval/data/direct/lcb.jsonl",
    "verilog_eval_v1-human": "fs_eval/data/direct/verilogeval-v1-human.jsonl",
    "verilog_eval_v1-machine": "fs_eval/data/direct/verilogeval-v1-machine.jsonl",
}


support_dataset = list(dataset_problem_map.keys())
