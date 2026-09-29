import dspy
import pickle
import os
from dotenv import load_dotenv

# Import our setup from the main script
from competition_skeleton import (
    GuardrailSignature,
    load_examples,
    split_examples,
    API_BASE
)

load_dotenv()
API_KEY = os.environ["LAB_API_KEY"]

# We only need the student model for inference
gemma = dspy.LM(
    "openai/google/gemma-4-E2B-it",
    api_base=API_BASE,
    api_key=API_KEY,
    max_tokens=1000,
)
dspy.configure(lm=gemma)

OUTPUT_PATH = "competition_guardrail"

def run_error_analysis():
    print("Loading data and compiled guardrail...")
    _, valset = split_examples(load_examples())

    file_path = f"{OUTPUT_PATH}/program.pkl"
    with open(file_path, "rb") as f:
        guardrail = pickle.load(f)

    failures = []

    print(f"Scanning {len(valset)} validation examples for errors...\n")

    for i, example in enumerate(valset):
        prediction = guardrail(image=example.image, prompt=example.prompt)

        true_label = str(example.label).strip().lower()
        pred_label = str(prediction.label).strip().lower()

        if true_label != pred_label:
            failures.append({
                "prompt": example.prompt,
                "true_label": true_label,
                "pred_label": pred_label,
                "reasoning": getattr(prediction, "reasoning", "No reasoning generated.")
            })

    print("=" * 60)
    print(f"ERROR REPORT: {len(failures)} failures out of {len(valset)}")
    print("=" * 60 + "\n")

    for idx, fail in enumerate(failures, 1):
        print(f"Failure {idx}/{len(failures)}")
        print(f"Prompt:    {fail['prompt']}")
        print(f"Expected:  {fail['true_label'].upper()}")
        print(f"Predicted: {fail['pred_label'].upper()}")
        print(f"Reasoning: {fail['reasoning']}")
        print("-" * 60)

if __name__ == "__main__":
    run_error_analysis()