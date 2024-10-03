SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR=${1}
MODEL_NAME=${2}
BENCHMARK=${3}
BATCH_EVAL=${4:-false}

set -x
set -e

python "${SCRIPTS_DIR}"/gen_answer_model.py --model_path $MODEL_DIR --model_name $MODEL_NAME --benchmark $BENCHMARK

CMD="python "${SCRIPTS_DIR}"/gen_judgment_batched_model.py --answer_model $MODEL_NAME --benchmark $BENCHMARK --judge_model "gpt-4o-mini-2024-07-18""
if [ "$BATCH_EVAL" = true ]; then
    CMD="$CMD --use_batch_api"
fi

$CMD

python "${SCRIPTS_DIR}"/show_result_model.py --answer_model $MODEL_NAME --judge_model "gpt-4o-mini-2024-07-18" --benchmark $BENCHMARK