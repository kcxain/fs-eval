from evaluator import dataset_input_map, dataset_problem_map
from utils.io import read_jsonl, write_jsonl


def mk_prompt_direct(question):
    system_prompt = """\
You are a Software Engineer. \
You need to write the corresponding Python code based on the given problem, and your output should only include the corresponding Python code enclosed in ```python ```.
For example:
## problem:
Given a list of numbers, propose code that returns the number of distinct elements in the list. Additionally, ensure that the solution has a time complexity of O(nlogn), where n is the length of the list.
The function definition of the code should be:
def count_distinct_elements(nums):
## answer:
```python
def count_distinct_elements(lst):
    if not lst:
        return 0
    lst.sort()
    count = 1
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            count += 1
    return count```

Now, please write the Python code based on the problem given by the user.
"""
    user_prompt = "## problem:\n" + question.strip() + "\n"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return conversation


def convert_to_sft(data_path):
    for task in read_jsonl(data_path):
        question = ""
        question += task["problem_description"].strip() + "\n"
        question += "The starter code should be:\n"
        question += "```python\n" + task["starter_code"].strip() + "```" + "\n"
        question = mk_prompt_direct(question)
        yield {"question": question, "task_id": task["task_id"]}


if __name__ == "__main__":
    write_jsonl(
        convert_to_sft(dataset_problem_map["compute-eval"]),
        dataset_input_map["compute-eval"],
    )
