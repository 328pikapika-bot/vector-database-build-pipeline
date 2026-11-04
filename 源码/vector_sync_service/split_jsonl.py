import json
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "kafka_messages_error.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        output_path = os.path.join(OUTPUT_DIR, f"{i}.json")
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        print(f"[{i}/100] -> {output_path}")

print(f"\nDone! {i} files written to {OUTPUT_DIR}")
