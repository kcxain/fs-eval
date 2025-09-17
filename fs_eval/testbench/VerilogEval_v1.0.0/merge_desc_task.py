from pathlib import Path
import json

human_desc = (
    "fs_eval/testbench/VerilogEval_v1.0.0/descriptions/VerilogDescription_Human.jsonl"
)
machine_desc = (
    "fs_eval/testbench/VerilogEval_v1.0.0/descriptions/VerilogDescription_Machine.jsonl"
)
human_data = "fs_eval/testbench/VerilogEval_v1.0.0/data/VerilogEval_Human.jsonl"
machine_data = "fs_eval/testbench/VerilogEval_v1.0.0/data/VerilogEval_Machine.jsonl"
merged_human = (
    "fs_eval/testbench/VerilogEval_v1.0.0/data/VerilogEval_Human_Merged.jsonl"
)
merged_machine = (
    "fs_eval/testbench/VerilogEval_v1.0.0/data/VerilogEval_Machine_Merged.jsonl"
)


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(data, file_path):
    ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def write_json(data, file_path, indent=None):
    ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def merge_desc_task(desc_file, data_file, output_file):
    descs = read_jsonl(desc_file)
    desc_dict = {desc["task_id"]: desc for desc in descs}
    data = read_jsonl(data_file)
    for item in data:
        task_id = item["task_id"]
        if task_id in desc_dict:
            item.update(desc_dict[task_id])
    write_jsonl(data, output_file)


if __name__ == "__main__":
    merge_desc_task(human_desc, human_data, merged_human)
    merge_desc_task(machine_desc, machine_data, merged_machine)
