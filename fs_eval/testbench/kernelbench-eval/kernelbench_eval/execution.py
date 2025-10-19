import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional


import tempfile
import os
import glob
import re
import json
from numpy import short
import ray
import torch
from typing import *
import subprocess
from loguru import logger
from unittest.mock import patch
from kernelbench_eval.reward.refine import Refine, EvalContent

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "reward", "test_code_tmpl.py"), "r") as fin:
    TEST_CODE_TMPL = fin.read()
with open(os.path.join(current_dir, "reward", "profiler.py"), "r") as fin:
    PROFILER_CODE = fin.read()


def parse_eval_msg(msg: str, raw_msg: str) -> EvalContent:
    # 解析结果
    speedup_match = re.search(
        r"Torch time:\s*([\d.]+)s,\s*CUDA time:\s*([\d.]+)s,\s*Speedup:\s*([\d.]+)x",
        msg,
    )
    assert speedup_match is not None, f"cannot find speedup info from test log: {msg}"
    speed_up_dict = {}
    speed_up_dict["torch_time"] = float(speedup_match.group(1))
    speed_up_dict["cuda_time"] = float(speedup_match.group(2))
    speed_up_dict["speedup"] = float(speedup_match.group(3))

    torch_match = re.search(
        r"#### Benchmark Torch Start\s*(.*?)\s*#### Benchmark Troch End", msg, re.S
    )
    cuda_match = re.search(
        r"#### Benchmark CUDA Start\s*(.*?)\s*#### Benchmark CUDA End", msg, re.S
    )
    assert torch_match is not None, (
        f"cannot find torch benchmark info from test log: {msg}"
    )
    assert cuda_match is not None, (
        f"cannot find cuda benchmark info from test log: {msg}"
    )
    torch_profiler = torch_match.group(1).strip()
    cuda_profiler = cuda_match.group(1).strip()
    return EvalContent(
        error_msg=None,
        torch_profile={
            "torch": torch_profiler,
            "cuda": cuda_profiler,
        },
        torch_time=speed_up_dict["torch_time"],
        cuda_time=speed_up_dict["cuda_time"],
        speedup=speed_up_dict["speedup"],
    )


def _compile_ext(cuda_code: str) -> Tuple[bool, Dict]:
    ret = {
        "ext_filename": None,
        "ext_content": None,
        "msg": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)

        compile_log = ""
        success = True
        try:
            compile_cmd = "python model_new.py"
            # A100: 8.0
            with patch.dict(
                os.environ,
                {
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                    "TORCH_EXTENSIONS_DIR": "build",
                    "MAX_JOBS": "1",
                },
            ):
                compile_result = subprocess.run(
                    compile_cmd,
                    timeout=180,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], "rb") as fin:
                bin_content = fin.read()
            ret["ext_filename"] = os.path.basename(so_files[0])
            ret["ext_content"] = bin_content
            ret["msg"] = "compile success"
            success = True
        except subprocess.TimeoutExpired:
            success = False
            ret["msg"] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret["msg"] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret


def _exec_eval(
    ext_filename: str, ext_content: bytes, cuda_code: str, pytorch_module: str
) -> Tuple[bool, EvalContent]:
    """Compile and execute test code which checks output with cuda implementation and pytorch module
    :param ext_filename: the cuda extension filename, in the format as "cuda_module.cpython-xxx.so"
    :param ext_content: file content of the extension file
    :param cuda_code: file content of the python file containing inline cuda code
    :param pytorch_module: pytorch baseline implementation. Should have Model.forward(...) and get_inputs() api
    :return (status,msg): (True,stdout) for success, (False,stderr) for error
    """
    device = ray.get_gpu_ids()
    torch.cuda.is_available()
    logger.info(f"current device: {device}, ext_filename: {ext_filename}")
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ext_filename), "wb") as fout:
            fout.write(ext_content)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)
        with open(os.path.join(tmpdir, "model.py"), "w") as fout:
            fout.write(pytorch_module)
        with open(os.path.join(tmpdir, "test.py"), "w") as fout:
            fout.write(TEST_CODE_TMPL)
        # with open(os.path.join(tmpdir, "profiler.py"), "w") as fout:
        #     fout.write(PROFILER_CODE)

        test_log = ""
        try:
            test_cmd = "python test.py"
            test_result = subprocess.run(
                test_cmd,
                timeout=60,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = test_result.stdout.decode()
        except subprocess.TimeoutExpired:
            # 超时
            return False, EvalContent(error_msg="failed: test timed out")
        except Exception as e:
            # 错误
            return False, EvalContent(
                error_msg=f"failed: test error: [{e}] log: [{test_log}]"
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                # logger.info(
                #     f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                # )
        if test_result.returncode != 0:
            # assert error, 输出可能过长
            lines = test_log.splitlines()
            filtered = []
            for line in lines:
                if "AssertionError" in line or "Mismatch" in line:
                    filtered.append(line)
            short_log = "\n".join(filtered)
            return False, EvalContent(error_msg=f"failed: test error: {short_log}")
    assert "#### Correctness check passed!" in test_log
    return True, parse_eval_msg(test_log, test_log)


def _extract_cuda_code(text: str):
    codeblock_seps = ["python"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```({languages_pattern})"
    pattern = re.compile(codeblock_start + r"\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return code_content
    return None


def _validate_cuda_code(code: str):
    all_ops = set(torch.ops.aten.__dict__.keys())
    allowed_ops = set([
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "numel",
        "view",
        "copy",
        "dim",
        "eye",
        "full",
        "full_like",
        "mode",
        "new_empty",
        "new_empty_strided",
        "new_full",
        "new_ones",
        "new_zeros",
        "randn",
        "rand",
    ])
    forbidden_ops = all_ops - allowed_ops
    pattern = re.compile(
        pattern="(torch::|aten::|torch\.)(" + "|".join(forbidden_ops) + ")\(",
        flags=re.DOTALL,
    )
    matched = re.search(pattern, code)
    if matched is not None:
        return False, f"Using {matched.group(0)[:-1]} is forbidden"
    return True, "success"


def compute_score(
    solution_str, ground_truth, extra_info, config=None, *args, **kwargs
) -> Refine:
    op_name = extra_info["op_name"]
    cuda_code = solution_str

    if cuda_code is None:
        return Refine(
            code=cuda_code,
            ops_name=op_name,
            formated=False,
        )
    else:
        validate_ret, validate_msg = _validate_cuda_code(cuda_code)
        if not validate_ret:
            return Refine(
                code=cuda_code,
                ops_name=op_name,
                formated=True,
                compiled=False,
                passed=False,
                compile_msg=validate_msg,
            )
        remote_compile_ext = ray.remote(num_cpus=8)(_compile_ext)
        status, compile_res = ray.get(remote_compile_ext.remote(cuda_code=cuda_code))
        # 编译错误
        if not status:
            return Refine(
                code=cuda_code,
                ops_name=op_name,
                formated=True,
                compiled=False,
                passed=False,
                compile_msg=compile_res["msg"],
            )
        try:
            gt_dict = json.loads(ground_truth)
            pytorch_module = gt_dict["pytorch_module"]
        except:
            pytorch_module = ground_truth
        ext_filename = compile_res["ext_filename"]
        ext_content = compile_res["ext_content"]
        run_kwargs = dict(
            ext_filename=ext_filename,
            ext_content=ext_content,
            cuda_code=cuda_code,
            pytorch_module=pytorch_module,
        )
        gpu_eval_task = ray.remote(num_gpus=1)(_exec_eval)
        eval_future = gpu_eval_task.remote(**run_kwargs)
        status, eval_content = ray.get(eval_future)

        # 执行错误
        if not status:
            return Refine(
                code=cuda_code,
                ops_name=op_name,
                formated=True,
                compiled=True,
                passed=False,
                eval_msg=eval_content,
            )

        return Refine(
            code=cuda_code,
            ops_name=op_name,
            formated=True,
            compiled=True,
            passed=True,
            eval_msg=eval_content,
        )


# def compute_score_remote(problem: Dict, completion: str, timeout: float):
#     solution_str = completion
#     ground_truth = problem["ground_truth"]["pytorch_module"]
#     extra_info = problem["extra_info"]
#     res = compute_score(solution_str, ground_truth, extra_info)
#     if res.passed:
#         return "passed"
#     else:
#         if not res.formated:
#             return "not formatted"
#         elif not res.compiled:
#             return f"compile error: {res.compile_msg}"
#         else:
#             return f"test error: {res.eval_msg.error_msg}"


def compute_score_remote(problem: Dict, completion: str):
    import requests

    gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")

    solution_str = completion
    ground_truth = problem["ground_truth"]["pytorch_module"]
    # curl -X POST http://10.200.198.14:8000/compute_score   -F "code=123"   -F "timeout_sec=60"   -F "nvcc_flags=-O2 -arch=sm_80"
    response = requests.post(
        url=f"http://{gpu_ip}:8000/compute_score",
        data={
            "cuda_code": solution_str,
            "torch_code": ground_truth,
        },
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    res_json = response.json()
    print(res_json)
    if res_json["passed"]:
        return "passed"
    else:
        return res_json["msg"]


def check_correctness(
    problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    result = compute_score_remote(problem, completion)

    return dict(
        task_id=problem["task_id"],
        passed=result == "passed",
        result=result,
        completion_id=completion_id,
    )
