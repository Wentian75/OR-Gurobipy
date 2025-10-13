from datasets import load_dataset
import json, sys, os 

def main(out_path='or_instruct_3k.jsonl'):
    ds = load_dataset("CardinalOperations/OR-Instruct-Data-3K", split="train")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in ds:
            prompt = r.get("prompt") or r.get("instruction") or r.get("input")
            completion = r.get("completion") or r.get("response") or r.get("output")
            if prompt and completion:
                f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
    print("Wrote", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv)>1 else "or_instruct_3k.jsonl"
    main(out)