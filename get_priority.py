import requests
import json

url = "http://127.0.0.1:8888/score"
input_file = "alpaca_requests_1000.jsonl"
output_file = "alpaca_requests_1000_priority.jsonl"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for i, line in enumerate(fin, start=1):
        prompt = line.strip()
        obj = json.loads(line)
        if not prompt:
            continue  # skip empty lines

        print(f"Request {i}: {prompt}")

        response = requests.post(url, json={"prompt": obj["prompt"]})
        fout.write(f"{response.text}\n")

print(f"✅ All responses saved to {output_file}")