import pandas as pd
import argparse
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--parquet_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="None")
    args = argparser.parse_args()

    parquet_path = args.parquet_path
    assert parquet_path.endswith(".parquet")
    output_path = args.output_path
    if output_path == "None":
        output_path = parquet_path.replace(".parquet", ".jsonl")
    data = pd.read_parquet(parquet_path)
    data = data.to_dict(orient="records")
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Converted {parquet_path} to {output_path}")
