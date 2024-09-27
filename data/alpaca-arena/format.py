import json
import uuid

with open("original_questions.json") as f:
    data = json.load(f)

# old keys
# dataset, instruction, output, generator
# new keys
# question_id, category, cluster, turns{ [content] }

# we should map dataset -> cluster
# question_id is uuid1
# category is "alpaca-eval"
# turns: content is instruction

new_data = [
    {
        "question_id": str(uuid.uuid1()),
        "category": "alpaca-eval",
        "cluster": item["dataset"],
        "turns": [{"content": item["instruction"]}],
    }
    for item in data
]


with open("question.jsonl", "w") as f:
    f.write("\n".join([json.dumps(d) for d in new_data]))
    f.write("\n")
