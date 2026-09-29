import dspy
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from competition_skeleton import (
    GuardrailSignature,
    load_examples,
    split_examples,
    evaluate,
    API_BASE
)

load_dotenv()
API_KEY = os.environ["LAB_API_KEY"]

gemma = dspy.LM(
    "openai/google/gemma-4-E2B-it",
    api_base=API_BASE,
    api_key=API_KEY,
    max_tokens=1000,
)
dspy.configure(lm=gemma)

OUTPUT_PATH = "competition_guardrail"

def load_guardrail(path: str = OUTPUT_PATH) -> dspy.Module:
    """Loads the compiled DSPy module directly from the pickled program file."""
    file_path = f"{path}/program.pkl"
    print(f"Loading compiled guardrail from {file_path}...")

    with open(file_path, "rb") as f:
        guardrail = pickle.load(f)

    return guardrail

if __name__ == "__main__":
    _, valset = split_examples(load_examples())

    submission_guardrail = load_guardrail(OUTPUT_PATH)

    print("Running evaluation...")
    reproduced_accuracy = evaluate(submission_guardrail, valset)

    print(f"\nReproduced Accuracy: {reproduced_accuracy:.1f}%")