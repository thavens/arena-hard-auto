SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR=${1}
MODEL_NAME=${2}

python "${SCRIPTS_DIR}"/gen_answer_model.py --model_path $MODEL_DIR --model_name $MODEL_NAME
python "${SCRIPTS_DIR}"/gen_judgment_batched_model.py --answer_model $MODEL_NAME --judge_model "gpt-4o-mini-2024-07-18" # official version uses: "gpt-4-1106-preview"
python "${SCRIPTS_DIR}"/arena-hard-auto/show_result_model.py --answer_model $MODEL_NAME --judge_model "gpt-4o-mini-2024-07-18"
