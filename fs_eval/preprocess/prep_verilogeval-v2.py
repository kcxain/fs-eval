from utils.io import read_jsonl, write_jsonl
from evaluator import dataset_input_map, dataset_problem_map
from utils.io import read_jsonl, write_jsonl

"""
TODO
FS-Coder
- non-think: direct
- think: cot
"""


def mk_prompt_direct_verilog(question):
    question = question.replace(
        "Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n",
        "",
    )
    system_prompt = """\
You are a Hardware Engineer. \
You need to write the corresponding Verilog code based on the given problem, and your output should only include the corresponding Verilog code enclosed in ```verilog ```.
For example:
## problem:
Design a 4-bit synchronous down counter that decrements by 1 on each positive edge of the clock signal. When the reset signal is active high, the counter should be reset to its maximum value.
The module head of the code should be:
```verilog
module fbitsyncdw(clk,rst,count);
```
## answer:
```verilog
module fbitsyncdw(clk,rst,count);
input clk,rst;
output reg[3:0]count;
always @(posedge clk or posedge rst ) begin
    if(rst)
    count<=4'hF;
    else
    count<=count-1'b1;
end
endmodule```

Now, please write the Verilog code based on the problem given by the user.
"""
    user_prompt = "## problem:\n" + question.strip() + "\n"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return conversation


def convert_to_sft_v1(data_path, description_path, prompt_type):
    tasks = list(read_jsonl(data_path))
    descriptions = list(read_jsonl(description_path))
    tasks.sort(key=lambda item: item["task_id"])
    descriptions.sort(key=lambda item: item["task_id"])
    for task, description in zip(tasks, descriptions):
        question = (
            description["detail_description"]
            + "The module head should be:\n"
            + task["prompt"]
        )
        if prompt_type == "simple":
            question = question.replace(
                "Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n",
                "",
            )

        question = mk_prompt_direct_verilog(question)

        yield {"question": question, "task_id": task["task_id"]}


def convert_to_sft_v2(data_path, prompt_type):
    for task in read_jsonl(data_path):
        question = task["fullprompt"]
        if question.count("//") >= 2 and "module TopModule" in question:
            # newly added
            question = question.split("//")
            question[-1] = "The module head should be:" + question[-1]
            question = "".join(question)
            # question = "```verilog\n" + question + "```"
        if prompt_type == "simple":
            question = question.replace(
                "Enclose your code with [BEGIN] and [DONE]. Only output the code snippet\nand do NOT output anything else.\n\n",
                "",
            )

        question = mk_prompt_direct_verilog(question)

        yield {"question": question, "task_id": task["task_id"]}


if __name__ == "__main__":
    write_jsonl(
        convert_to_sft_v2(dataset_input_map["verilog_eval_v1-machine"]),
        dataset_problem_map["verilog_eval_v1-machine"],
    )
