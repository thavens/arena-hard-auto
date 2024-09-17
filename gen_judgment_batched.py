import argparse
import os
import re
import json

from utils import (
    load_questions,
    load_model_answers,
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


def get_judge_prompt(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id": question["question_id"],
        "model": answer["model_id"],
        "judge": configs["judge_model"],
        "games": [],
        "output_file": args["output_file"],
    }

    convs = []
    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
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

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]

            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        convs.append(conv)
    output["convs"] = convs
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(
        f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
        + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}'
    )

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)

    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]

    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]

    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]

    judgments = []  # list of dict by question id, conv key is pair of convs
    print("creating conversations")
    for model in models:
        count = 0
        for question in questions:
            question_id = question["question_id"]

            kwargs = {}
            kwargs["question"] = question
            if model in model_answers and question_id not in model_answers[model]:
                print(
                    f"Warning: {model} answer to {question['question_id']} cannot be found."
                )
                continue

            if model in existing_judgments and question_id in existing_judgments[model]:
                count += 1
                continue

            kwargs["answer"] = model_answers[model][question_id]
            if ref_answers:
                kwargs["reference"] = [
                    ref_answer[question_id] for ref_answer in ref_answers
                ]
                assert len(kwargs["reference"]) == len(configs["ref_model"])
            else:
                kwargs["reference"] = None
            if configs["baseline"]:
                kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][
                    question_id
                ]
            else:
                kwargs["baseline_answer"] = None
            kwargs["configs"] = configs
            kwargs["endpoint_dict"] = endpoint_info
            kwargs["output_file"] = output_files[model]
            kwargs["regex_pattern"] = pattern
            output_objects = get_judge_prompt(**kwargs)
            judgments.append(output_objects)

        print(f"{count} number of existing judgments")

    print("sending to batched API")
    flattened_list = [conv for judgment in judgments for conv in judgment["convs"]]
    api_dict = get_endpoint(endpoint_info["endpoints"])
    responses_flattened = batch_api_call(
        flattened_list,
        endpoint_info["model_name"],
        configs["temperature"],
        configs["max_tokens"],
        api_dict,
    )
    judgments = match_responses(
        judgments=judgments, responses_flattened=responses_flattened
    )

    print("writing outputs")
    for output_object in judgments:
        for game, (conv, response) in enumerate(
            zip(output_object["convs"], output_object["responses"])
        ):
            score, failed = get_score(response, pattern, configs["pairwise"])
            result = {
                "user_prompt": conv[1]["content"],
                "judgment": response,
                "score": "A=B" if failed else score,
            }
            output_object["games"].append(result)

        with open(output_object["output_file"], "a") as f:
            del output_object["convs"]
            del output_object["responses"]
            del output_object["output_file"]
            f.write(json.dumps(output_object, ensure_ascii=False) + "\n")
