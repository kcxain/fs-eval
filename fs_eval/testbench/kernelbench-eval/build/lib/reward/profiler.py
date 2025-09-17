import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import functools
import os


def torch_profiler(print_table=True):
    def decorator(forward_func):
        @functools.wraps(forward_func)
        def wrapper(self, *args, **kwargs):
            # 检测设备
            device = "cpu"
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_cuda:
                    device = "cuda"
                    break

            # 配置profiler活动
            activities = [ProfilerActivity.CPU]
            if device == "cuda":
                activities.append(ProfilerActivity.CUDA)

            # 使用profiler
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                with record_function(f"{self.__class__.__name__}_forward"):
                    result = forward_func(self, *args, **kwargs)

            # 打印结果
            if print_table:
                sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
                print("#### Torch Profiler Begin ####")
                print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
                print("#### Torch Profiler End ####")

            return result

        return wrapper

    return decorator
