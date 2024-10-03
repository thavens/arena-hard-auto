"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import shortuuid

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput

from utils import (
    load_questions,
    reorg_answer_file,
)


def question_to_conv(question):
    conv = []
    conv.append({"role": "system", "content": "You are a helpful assistant."})

    assert (
        len(question["turns"]) == 1
    ), "Local currently doesn't support multi turn. default question data should have been only single turn"
    conv.append({"role": "user", "content": question["turns"][0]["content"]})

    return conv


def gen_answers(
    model: str,
    model_id: str,
    questions: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    num_choices: int = 1,
) -> list:
    """generate answers for 1 model without caching for arena hard auto

    Args:
        model (str): path to hf model folder
        temperature (float, optional): Defaults to 0.0.
        max_tokens (int, optional): max_tokens vllm. Defaults to 4096.
        num_choices (int, optional): number of times to sample an output. Defaults to 1.

    Returns:
        list: jsonl of answers in format for arena hard auto
    """
    convs = []
    for question in questions:
        for i in range(num_choices):
            convs.append(question_to_conv(question))

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count(), max_model_len=max_tokens)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
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
    for question in questions:
        choices = []
        for i in range(num_choices):
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
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    args = parser.parse_args()

    base_folder = Path(__file__).absolute().parent
    os.makedirs(base_folder / "data" / args.benchmark / "answers", exist_ok=True)
    answer_file = (
        base_folder / "data" / args.benchmark / "answers" / f"{args.model_name}.jsonl"
    )

    base_folder = Path(__file__).absolute().parent
    question_file = base_folder / "data" / args.benchmark / "question.jsonl"

    questions = load_questions(question_file)
    answers = gen_answers(args.model_path, args.model_name, questions)

    with open(answer_file, "w") as fout:
        answers_str = [json.dumps(ans) for ans in answers]
        fout.write("\n".join(answers_str))
        fout.write("\n")
