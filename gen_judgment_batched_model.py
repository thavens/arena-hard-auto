import argparse
import os
import re
import json
from pathlib import Path

from utils import (
    load_questions,
    make_config,
    batch_api_call,
    match_responses,
    get_endpoint,
)


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        print("No regex match")
        print(judgment)
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        print("Multiple regex match")
        return None, True

system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
template = "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"

def get_judge_prompt(question, answer, baseline, pairwise = True):
    num_games = 2 if pairwise else 1

    output = {
        "model": answer["model_id"],
        "question_id": question["question_id"],
        "games": [],
    }

    convs = []
    for game in range(num_games):
        conv = [{"role": "system", "content": system_prompt}]

        prompt_args = {}

        for i, turn in enumerate(question["turns"]):
            prompt_args[f"question_{i+1}"] = turn["content"]
        base = 1

        if baseline:
            if game % 2 == 1:  # swap position
                answer, baseline = baseline, answer

            for i, turn in enumerate(baseline["choices"][0]["turns"]):
                prompt_args[f"answer_{i+1}"] = turn["content"]
                base += 1
        if answer:
            for i, turn in enumerate(answer["choices"][0]["turns"]):
                prompt_args[f"answer_{i+base}"] = turn["content"]

        user_prompt = template.format(**prompt_args)
        conv.append({"role": "user", "content": user_prompt})

        convs.append(conv)
    output["convs"] = convs
    return output


# keys: question_id, model, judge, games [{user_prompt, judgment, score}]
def batch_eval_answers(answers: list, judge_model: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.0, max_tokens: int = 4096):
    base_folder = Path(__file__).absolute().parent
    pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
    question_file = base_folder / "data" / "arena-hard-v0.1" / "question.jsonl"
    questions = load_questions(question_file)
    with open(base_folder / "data" / "arena-hard-v0.1" / "model_answer" / "gpt-4-0314.jsonl") as f:
        baseline_answers = [json.loads(line) for line in f]
    
    baseline_answers = {answer["question_id"]: answer for answer in baseline_answers}
    model_answers = {answer["question_id"]: answer for answer in answers}
    
    judgments = []  # list of dict by question id, conv key is pair of convs
    print("creating conversations")
    for question in questions:
        question_id = question["question_id"]
        output_objects = get_judge_prompt(question, model_answers[question_id], baseline_answers[question_id])
        judgments.append(output_objects)

    flattened_list = [conv for judgment in judgments for conv in judgment["convs"]]
    responses_flattened = batch_api_call(
        flattened_list,
        judge_model,
        temperature,
        max_tokens,
    )
    judgments = match_responses(
        judgments=judgments, responses_flattened=responses_flattened
    )
    
    for output_object in judgments:
        for conv, response in zip(output_object["convs"], output_object["responses"]):
            score, failed = get_score(response, pattern)
            result = {
                "user_prompt": conv[1]["content"],
                "judgment": response,
                "score": "A=B" if failed else score,
            }
            output_object["games"].append(result)
    
    for j in judgments:
        del j["convs"]
        del j["responses"]
    return judgments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", type=str)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini-2024-07-18", choices=["gpt-4o-mini-2024-07-18", "gpt-4-1106-preview"])
    args = parser.parse_args()
    print(args)

    with open(args.answer_file) as f:
        answers = [json.loads(line) for line in f]
    judgments = batch_eval_answers(answers, args.judge_model)
    
    with open("judge_output.jsonl", "w") as f:
        f.write("\n".join([json.dumps(judgment) for judgment in judgments]))
        f.write("\n")