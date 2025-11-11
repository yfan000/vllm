import requests
import json

url = "http://127.0.0.1:8888/score"
input_file = "alpaca_requests_1000.jsonl"
output_file = "alpaca_requests_1000_priority.jsonl"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for i, line in enumerate(fin, start=1):
        if not line.strip():
            continue  # skip empty lines
        obj = json.loads(line)
        prompt = obj["prompt"]
        

        response = requests.post(url, json={"prompt": prompt})
        # fout.write(f"{response.text}\n")
        priority = int(float(json.loads(response.text)["priority"]) * 1000000)
        record = {
            "prompt": prompt,
            "priority": priority
        }
        fout.write(json.dumps(record) + "\n")

print(f"✅ All responses saved to {output_file}")