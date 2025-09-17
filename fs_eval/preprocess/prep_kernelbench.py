from datasets import load_dataset, Dataset
import pandas as pd
from typing import Optional
from utils.io import write_jsonl


system_prompt = """\
You are an expert in PyTorch and CUDA programming. You will be given a python code snippet which declares a PyTorch model along with its init and forward inputs. The model is an instance of class `Model`. It will be created with arguments from `get_init_inputs()`. Then its `forward` function will be called with data from `get_inputs()`. Your task is to write a custom CUDA extension to accelerate the model `forward` function. Note that:
- Provide only a single python code block in your final answer.
- Name your optimized model as `ModelNew`. Keep its `__init__` and `forward` function signature the same as `Model`. Keep the names of all submodules unchanged. Ensure the keys of model state dict are unchanged. Do not create any extra tensor parameter during model initialization.
- Inline the CUDA code within quotes and assign it to the `source` variable. Inline the C++ function definition into the `cpp_src` variable. Compile and load the extension using `torch.utils.cpp_extension.load_inline`.
- Carefully decide the kernel function signature and pass the correct arguments into your kernel.
- Do not perform extra initialization on parameters of any submodule. Keep them initialized by default.
- Implement all CUDA operators by yourself. Do not call any function from `torch` namespace except for allocating or initializing tensors. Do not call any function from `torch.nn.functional` namespace. Do not call the forward function of any submodule. You can only use the parameters and attributes of the submodule. For example, you should pass `self.linear.weight` and `self.linear.bias` as arguments to your CUDA kernel instead of directly running `self.linear(x)`.
- You can implement more than one kernel in the CUDA extension. If there are multiple operators within `forward` function, you must implement all of them no matter how many CUDA kernels are needed.
For example:
## problem:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [2.0]
```
## answer:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
source = \"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = alpha * a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b, float alpha) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
\"""

cpp_src = \"""
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b, float alpha);
\"""

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return elementwise_add.elementwise_add_cuda(a, b, self.alpha)
```
Now, please write the PyTorch code with custom CUDA extension based on the problem given by the user.
"""


def get_sample(
    pytorch_module: str,
    dataset_id: str,
    level_id: str,
    pytorch_functional: Optional[str] = None,
    cuda_code: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> str:
    output = {
        "task_id": level_id + "-" + str(extra_info["problem_id"]),
        "question": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "## problem: "
                + "```python\n"
                + pytorch_module
                + "```"
                + "\n",
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
            "pytorch_functional": pytorch_functional or "",
            "cuda_code": cuda_code or "",
        },
        "style": "cuda-sandbox-v2",
        "extra_info": extra_info or {},
    }
    return output


def convert_kernel_bench():
    dataset_id = "ScalingIntelligence/KernelBench"
    dataset_full: dict[str, Dataset] = load_dataset(dataset_id)

    def convert_sample(sample):
        pytorch_module = sample["code"]
        level_id = str(sample["level"])
        op_name = sample["name"]
        problem_id = sample["problem_id"]
        output = get_sample(
            pytorch_module=pytorch_module,
            dataset_id=dataset_id,
            level_id=level_id,
            extra_info={"op_name": op_name, "problem_id": problem_id},
        )
        return pd.Series(output)

    for key, dataset in dataset_full.items():
        output_path = f"fs_eval/data/KernelBench-{key}.jsonl"
        df = dataset.to_pandas()
        df = df.apply(convert_sample, axis=1)
        df_json = df.to_dict(orient="records")
        write_jsonl(df_json, output_path)


if __name__ == "__main__":
    convert_kernel_bench()
