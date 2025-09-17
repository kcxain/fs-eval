from evaluator import dataset_input_map, dataset_problem_map
from utils.io import read_jsonl, write_jsonl


def mk_prompt_direct(question):
    system_prompt = """\
You are a CUDA programming expert capable of generating high-quality, efficient CUDA code with best practices, optimized for performance and clarity.
Instructions:
1. Implement the body of the function(s). Do not include any additional code outside the function(s).
2. Wrap the completed function code, including the provided signatures, inside a single ```cuda markdown code block.
3. Make sure to use the precise function signature in your response if it's provided in the query.

For example:
## problem:
Implement a function called `launch` that launches a kernel function named `kernel` with the provided grid and block dimensions using triple chevrons. The x,y,z grid sizes and block sizes will be provided as parameters\nto the `launch` function. Assume that the `kernel` function is already defined.
The signature of the `kernel` function is\n```cuda\n__global__ void kernel(int *output, const int *input) \n```
The function signature is \n```cuda\nvoid launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1, int gridSizeZ = 1, int blockSizeZ = 1)```

The following headers are already defined and should not be included in the response:
```cuda
#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
```
Please do not include any additional headers in your response.

## answer:
```cuda
void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, int gridSizeZ, int blockSizeZ) {
    dim3 blockSize(blockSizeX, blockSizeY, blockSizeZ); 
    dim3 gridSize(gridSizeX, gridSizeY, gridSizeZ);

    // Assuming 'output' and 'input' are already defined and allocated on the device
    int *output; 
    const int *input;

    // Launch the kernel with the provided grid and block dimensions
    kernel<<<gridSize, blockSize>>>(output, input);
}
```
"""
    user_prompt = "## problem:\n" + question.strip() + "\n"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return conversation


def extract_header_files_from_problem(problem) -> str:
    declaration = problem.get("declaration", "")
    header_files = []
    for header_file in declaration.split("\n"):
        header_file = header_file.strip()
        # header file starts with #include
        if header_file.startswith("#include"):
            header_files.append(header_file)
    return "\n".join(header_files)


HEADER_FILES_PROMPT_TEMPLATE = """\
The following headers are already defined and should not be included in the response:
```cuda
{header_files}
```
Please do not include any additional headers in your response.
"""


def convert_to_sft(data_path, prom):
    for task in read_jsonl(data_path):
        # for humaneval
        question = ""
        question += task["prompt"].strip() + "\n\n"
        header_files = extract_header_files_from_problem(task)
        question += HEADER_FILES_PROMPT_TEMPLATE.format(header_files=header_files)
        question = mk_prompt_direct(question)
        yield {"question": question, "task_id": task["task_id"]}


if __name__ == "__main__":
    write_jsonl(
        convert_to_sft(dataset_problem_map["compute-eval"]),
        dataset_input_map["compute-eval"],
    )
