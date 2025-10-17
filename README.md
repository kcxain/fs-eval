# FS-Eval

Eval full-stack code generation.


## Supported Benchmarks

### Verilog

- ✅ verilog_eval_v1-human
- ✅ verilog_eval_v1-machine
- [ ] RTLLM

### Python
- HumanEval
- [ ] MBPP

### Kernel
- Kernelbench
- Compute-Eval


## How to use

```bash
python -m fs_eval.sampler \
    --dataset kernelbench \
    --model_path Qwen2.5-Coder-7B-Instruct \
    --model_name Qwen2.5-Coder-7B-Instruct \
    --max_tokens 20480 \
    --temperature 0.6 \
    --sample_n 1 \
    --prompt_type cot \
    --top_p 0.95 \
    --top_k 20
```

## How to add a new benchmark

- add system_prompts and user_prompts to `fs-eval/fs_eval/prompts`
- register prompt builder to `fs-eval/fs_eval/prompt_builder.py`
- register evaluator to `fs-eval/fs_eval/evaluator.py`