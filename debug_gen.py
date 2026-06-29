import json
from pathlib import Path

from local_inference import generate, load_model
from scheduling_format import parse_makespan_completion
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

LORA_DIR = Path(__file__).parent / "lora_output"


def run_row(row: dict, name: str, model, tok) -> None:
    raw = generate(model, tok, row["description"], max_length=MAX_SEQ_LENGTH)
    pred = parse_makespan_completion(raw)
    print(
        name,
        "label",
        row["optimal_action"]["makespan"],
        "raw",
        repr(raw),
        "pred",
        pred,
    )


def main() -> None:
    model, tok = load_model(mode="lora", model_name=MODEL_NAME, lora_dir=LORA_DIR)
    train_row = json.loads(Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
    test_row = json.loads(Path("test_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
    run_row(train_row, "train", model, tok)
    run_row(test_row, "test", model, tok)


if __name__ == "__main__":
    main()
