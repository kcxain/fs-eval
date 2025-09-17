system_prompt = """You are a CUDA programming expert capable of generating high-quality, efficient CUDA code with best practices, optimized for performance and clarity."""

user_prompt = """You will be given an instruction, and you should complete the code. Note that:
- Think carefully before providing your final answer.
- Implement the body of the function(s). Do not include any additional code outside the function(s).
- Wrap the completed function code, including the provided signatures, inside a single ```cuda markdown code block.
- Make sure to use the precise function signature in your response if it's provided in the query.

Now you are given the below instruction:

__CUDA_PROBLEM__

Please respond with thinking process and your final answer.
"""


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
The following headers are already defined and should not be included in the answer:
```cuda
{header_files}
```
Please do not include any additional headers in your answer.
"""

user_prompt_oneshot = """You will be given an instruction, and you should complete the code. Note that:
- Think carefully before providing your final answer.
- Implement the body of the function(s). Do not include any additional code outside the function(s).
- Wrap the completed function code, including the provided signatures, inside a single ```cuda markdown code block.
- Make sure to use the precise function signature in your response if it's provided in the query.

Here is an basic example:

===== Example Start =====
Given instruction:

Implement a function called `launch` that launches a kernel function named `kernel` with the provided grid and block dimensions using triple chevrons. The x,y,z grid sizes and block sizes will be provided as parameters\nto the `launch` function. Assume that the `kernel` function is already defined.
The signature of the `kernel` function is\n```cuda\n__global__ void kernel(int *output, const int *input) \n```
The function signature is \n```cuda\nvoid launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1, int gridSizeZ = 1, int blockSizeZ = 1)```

The following headers are already defined and should not be included in the response:
```cuda
#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
```
Think carefully before providing your final answer and please do not include any additional headers in your answer.

You should respond with:
<think>
[Your thinking process]
</think>
Then the final answer:

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
===== Example End =====

Now you are given the below instruction:

__CUDA_PROBLEM__

Please respond with thinking process and your final answer.
"""


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
The following headers are already defined and should not be included in the answer:
```cuda
{header_files}
```
Please do not include any additional headers in your answer.
"""
