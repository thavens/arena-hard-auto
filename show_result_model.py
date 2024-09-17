import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils_math import (
    compute_mle_elo,
    get_bootstrap_result,
    get_win_rate_column,
)


def get_battles_from_row(
    row, first_game_only, multiplier, baseline_model, metadata=None
):
    results = []
    output = {
        "question_id": row["question_id"],
        "model_a": baseline_model,
        "model_b": row["model"],
    }

    game = row["games"][0]
    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_a"
    elif game["score"] == "A>>B":
        output["winner"] = "model_a"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_b"
    elif game["score"] == "B>>A":
        output["winner"] = "model_b"
        weight = multiplier
    else:
        weight = 0

    # add conv_metadata for style control
    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]][
                "conv_metadata"
            ]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]][
                "conv_metadata"
            ]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"][
                "list_count"
            ],
            "bold_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"][
                "bold_count"
            ],
        }

    if weight:
        results += [output] * weight

    if first_game_only:
        return results

    # game 2
    output = {
        "question_id": row["question_id"],
        "model_a": baseline_model,
        "model_b": row["model"],
    }

    game = row["games"][1]

    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_b"
    elif game["score"] == "A>>B":
        output["winner"] = "model_b"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_a"
    elif game["score"] == "B>>A":
        output["winner"] = "model_a"
        weight = multiplier
    else:
        weight = 0

    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]][
                "conv_metadata"
            ]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]][
                "conv_metadata"
            ]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"][
                "list_count"
            ],
            "bold_count_a": metadata[baseline_model][row["question_id"]][
                "conv_metadata"
            ]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"][
                "bold_count"
            ],
        }

    if weight:
        results += [output] * weight

    return results


def get_battles_from_judgment(
    judgments: list[dict],
    first_game_only=False,
    multiplier=3,
    baseline_model="gpt-4-0314",
):
    print("Turning judgment results into battles...")

    judgments = pd.DataFrame(judgments)

    metadata = None

    battles = judgments.apply(
        lambda row: get_battles_from_row(
            row, first_game_only, multiplier, baseline_model, metadata
        ),
        axis=1,
    )
    battles = pd.DataFrame(battles[battles.map(len) > 0].explode().tolist())
    return battles


def show_result(
    judgments, model_answers, baseline="gpt-4-0314", weight=3, num_rounds=100
):
    battles = get_battles_from_judgment(judgments, False, weight, baseline)

    bt_model_coef = compute_mle_elo(battles, baseline_model=baseline)
    bootstrap_model_coef = get_bootstrap_result(
        battles, compute_mle_elo, num_rounds, baseline
    )

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bt_model_coef.index):
        assert model in bootstrap_model_coef.columns
        stats.at[i, "model"] = model
        stats.at[i, "score"] = bt_model_coef[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_model_coef[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_model_coef[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                if "token_len" in turn:
                    length += turn["token_len"]
                else:
                    length += row["conv_metadata"]["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_model_coef[model].tolist()

    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(stats, "score", baseline).tolist()
    stats["lower"] = get_win_rate_column(stats, "lower", baseline).tolist()
    stats["upper"] = get_win_rate_column(stats, "upper", baseline).tolist()
    decimal = 1

    stats.sort_values(by="score", ascending=False, inplace=True)
    for _, row in stats.iterrows():
        interval = str(
            (
                round(row["lower"] - row["score"], decimal),
                round(row["upper"] - row["score"], decimal),
            )
        )
        print(
            f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file", type=str, default="arena-hard-v0.1")
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview.jsonl",
    )
    # parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    # parser.add_argument("--baseline", type=str, default="gpt-4-0314")
    # parser.add_argument("--load-bootstrap", action="store_true")
    # parser.add_argument("--show-elo", action="store_true")
    # parser.add_argument("--weight", type=int, default=3)
    # parser.add_argument("--num-rounds", type=int, default=100)
    # parser.add_argument("--first-game-only", action="store_true")
    args = parser.parse_args()
    print(args)
    # assert not args.load_bootstrap or (args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."
    base_folder = Path(__file__).absolute().parent
    with open(args.answer_file) as f:
        model_answers = [json.loads(line) for line in f]
        model_answers = {
            model_answers[0]["model_id"]: {i["question_id"]: i for i in model_answers}
        }

    with open(base_folder / "data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl") as f:
        ma = [json.loads(line) for line in f]
        model_answers[ma[0]["model_id"]] = {i["question_id"]: i for i in ma}

    with open(args.judge_file) as f:
        judgments = [json.loads(line) for line in f]

    show_result(judgments, model_answers)
