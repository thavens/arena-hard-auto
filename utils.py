import os
import json
import time
import yaml
import random
import requests
import io

from typing import Optional
from glob import glob

import dotenv
dotenv.load_dotenv()

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_openai_azure(
    model, messages, temperature, max_tokens, api_dict=None
):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2,
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
    ]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": message}]}],
                "safetySettings": safety_settings,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system": "SYSTEM", "assistant": "CHATBOT", "user": "USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append(
                {"role": template_map[message["role"]], "message": message["content"]}
            )
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break

    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


# {
#    "custom_id":"request-1",
#    "method":"POST",
#    "url":"/v1/chat/completions",
#    "body":{
#       "model":"gpt-3.5-turbo-0125",
#       "messages":[
#          {
#             "role":"system",
#             "content":"You are a helpful assistant."
#          },
#          {
#             "role":"user",
#             "content":"Hello world!"
#          }
#       ],
#       "max_tokens":1000
#    }
# }
# {
#    "custom_id":"request-2",
#    "method":"POST",
#    "url":"/v1/chat/completions",
#    "body":{
#       "model":"gpt-3.5-turbo-0125",
#       "messages":[
#          {
#             "role":"system",
#             "content":"You are an unhelpful assistant."
#          },
#          {
#             "role":"user",
#             "content":"Hello world!"
#          }
#       ],
#       "max_tokens":1000
#    }
# }
def conv_jsonl_to_batch_format(
    convs: list[list], model: str, temp: float, max_tok: int
):
    request = []
    for idx, conv in enumerate(convs):
        request.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": conv,
                    "temperature": temp,
                    "max_tokens": max_tok,
                },
            }
        )
    return request


#  endpoint_info["model_name"], configs["temperature"], configs["max_tokens"]
def batch_api_call(
    convs: list[list], model: str, temp: float, max_tok: int, api_dict=None, shortcut=""
):
    if len(convs) == 0:
        return []

    import openai

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    if shortcut:
        batch = client.batches.retrieve(shortcut)
    else:
        requests = conv_jsonl_to_batch_format(convs, model, temp, max_tok)

        input_str = "\n".join([json.dumps(requests) for requests in requests])
        file = io.BytesIO(bytes(input_str, "utf-8"))
        batch_input_file = client.files.create(file=file, purpose="batch")

        batch_input_file_id = batch_input_file.id

        batch_object = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Arena hard auto judge job"},
        )

        status_bits = {
            "validating": False,
            "failed": False,
            "in_progress": False,
            "finalizing": False,
            "completed": False,
            "expired": False,
            "cancelling": False,
            "cancelled": False,
        }

        while True:
            time.sleep(5)
            batch = client.batches.retrieve(batch_object.id)
            print(
                f"{batch.request_counts.completed} / {batch.request_counts.total}",
                end="\r",
            )
            if not status_bits[batch.status]:
                print(f"Batch status: {batch.status}")
                status_bits[batch.status] = True
            if batch.status == "completed":
                break
            if (
                batch.status == "cancelled"
                or batch.status == "failed"
                or batch.status == "expired"
            ):
                raise (Exception(f"Batch processing was {batch.status}"))

        print("waiting on file id")
        while batch.output_file_id is None:
            time.sleep(1)
            batch = client.batches.retrieve(batch_object.id)

    output = batch.output_file_id
    error_out = batch.error_file_id

    file_response = client.files.content(output)
    if error_out is not None:
        file_response_error = client.files.content(error_out)
        if file_response_error.text:
            print(f"**API ERRORS**\n{file_response_error.text}")
    jsonl = [json.loads(line) for line in file_response.text.split("\n") if line]
    request_id_2_response = {l["custom_id"]: l for l in jsonl}
    
    # the order is baseline first, then answer after. Second game is answer first baseline second
    # even index is baseline first, odd index is answer first
    # if index even, and answer is missing then A>B
    # if index odd, and answer is missing then A>B
    judgments = []
    for i in range(len(convs)):
        if f"{i}" not in request_id_2_response and i % 2 == 0:
            judgments.append("[[A>B]]")
        elif f"{i}" not in request_id_2_response and i % 2 == 1:
            judgments.append("[[A<B]]")
        else:
            judgments.append(request_id_2_response[str(i)]["response"]["body"]["choices"][0]["message"]["content"])
            
    return judgments


def match_responses(judgments, responses_flattened):
    # Keep track of the current position in the responses_flattened list
    index = 0

    # Iterate over each judgment dictionary
    new_judgments = []
    for judgment in judgments:
        convs_len = len(
            judgment["convs"]
        )  # Get the number of items in the "convs" list

        # Extract the corresponding number of responses from the responses_flattened list
        new_judgment = judgment.copy()
        new_judgment["responses"] = responses_flattened[index : index + convs_len]
        new_judgments.append(new_judgment)

        # Move the index forward by the number of responses we just added
        index += convs_len
    assert index == len(
        responses_flattened
    ), f"diff: {index}, {len(responses_flattened)}"
    return new_judgments
