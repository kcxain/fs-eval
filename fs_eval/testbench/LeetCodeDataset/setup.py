import os
from setuptools import setup, find_packages


def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="leetcodedataset",
    version="0.1.1",
    description="LeetCodeDataset",
    # 移除 py_modules=["leetcodedataset"]
    packages=["eval_lcd"],  # 明确指定包
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "eval_lcd = eval_lcd.evaluate:cli",
        ]
    },
    python_requires=">=3.6",
)
