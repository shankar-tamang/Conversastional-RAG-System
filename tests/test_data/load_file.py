import json

with open("tests/test_data/qa_pairs.json") as f:
    data = json.load(f)

print(data)