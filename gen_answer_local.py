"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""

import argparse
import json
import os
import re
import time
import concurrent.futures
from pathlib import Path

import tiktoken
import shortuuid
import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput

from add_markdown_info import count_markdown_elements, remove_pattern
from utils import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)


def question_to_conv(question):
    conv = []
    conv.append({"role": "system", "content": "You are a helpful assistant."})

    assert (
        len(question["turns"]) == 1
    ), "Local currently doesn't support multi turn. default question data should have been only single turn"
    conv.append({"role": "user", "content": question["turns"][0]["content"]})

    return conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    existing_answer = load_model_answers(
        os.path.join("data", settings["bench_name"], "model_answer")
    )

    print(settings)

    for model in settings["model_list_local"]:
        question_file = os.path.join("data", settings["bench_name"], "question.jsonl")
        questions = load_questions(question_file)

        model_id = Path(model).name
        answer_file = os.path.join(
            "data", settings["bench_name"], "model_answer", f"{model_id}.jsonl"
        )
        print(f"Output to {answer_file}")

        count = 0
        convs = []
        for question in questions:
            if (
                model_id in existing_answer
                and question["question_id"] in existing_answer[model_id]
            ):
                count += 1
                continue
            for i in range(settings["num_choices"]):
                convs.append(question_to_conv(question))

        if count > 0:
            print(f"{count} number of existing answers")
            if count == len(questions):
                print(f"{count} answers == {len(questions)} questions. skipping eval for {model}")
                continue

        tokenizer = AutoTokenizer.from_pretrained(model)
        llm = LLM(model=model, tensor_parallel_size=2)
        sampling_params = SamplingParams(
            temperature=settings["temperature"], max_tokens=settings["max_tokens"]
        )
        prompts = tokenizer.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=False
        )

        responses: list[RequestOutput] = llm.generate(prompts, sampling_params)

        responses = [
            {
                "content": output.outputs[0].text,
                "token_len": len(output.outputs[0].token_ids),
            }
            for output in responses
        ]

        curr_response = 0
        answers = []
        for index, question in enumerate(questions):
            if (
                model_id in existing_answer
                and question["question_id"] in existing_answer[model_id]
            ):
                continue

            choices = []
            for i in range(settings["num_choices"]):
                choices.append({"index": i, "turns": [responses[curr_response]]})
                curr_response += 1

            ans = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            answers.append(ans)

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            answers_str = [json.dumps(ans) for ans in answers]
            fout.write("\n".join(answers_str))
            fout.write("\n")

        reorg_answer_file(answer_file)
