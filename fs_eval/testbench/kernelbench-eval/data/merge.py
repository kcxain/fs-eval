from utils.io import read_jsonl, write_jsonl

data_paths = [
    "/share/personal/I/qimeng6/fs-coder/eval/testbench/kernelbench-eval/data/KernelBench-level_1.jsonl",
]

merg_data = []
for path in data_paths:
    data = read_jsonl(path)
    for d in data:
        d["task_id"] = d["level"] + "-" + str(d["extra_info"]["problem_id"])
        merg_data.append(d)
write_jsonl(merg_data, "fs_eval/testbench/kernelbench-eval/data/KernelBench-level_1.jsonl")
