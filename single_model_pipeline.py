import argparse
import json
from pathlib import Path

from gen_answer_model import gen_answers
from gen_judgment_batched_model import batch_eval_answers
from show_result_model import show_result

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--judge", type=str, default="gpt-4o-mini-2024-07-18", choices=["gpt-4o-mini-2024-07-18", "gpt-4-0314"])
args = parser.parse_args()

print(args)

answers = gen_answers(args.model_path)
judgments = batch_eval_answers(answers, args.judge)

base_folder = Path(__file__).absolute().parent
answers_converted = {}
answers_converted[answers[0]["model_id"]] = {i["question_id"]: i for i in answers}
with open(base_folder / "data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl") as f:
    ma = [json.loads(line) for line in f]
    answers_converted[ma[0]["model_id"]] = {i["question_id"]: i for i in ma}
show_result(judgments, answers_converted)

