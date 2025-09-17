import re


def extract_code_block(text: str):
    codeblock_seps = ["python", "cuda", "verilog"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```({languages_pattern})"
    pattern = re.compile(codeblock_start + r"\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return code_content
    return "CANNOT EXTRACT CODE"
