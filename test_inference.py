"""
Quick inference test on the GEKO-trained GPT-2 checkpoint.
Compares base GPT-2 vs GEKO-finetuned on math problems from OpenR1-Math-220k.
"""

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CHECKPOINT = "./geko_r1_output/checkpoint-48"
MAX_NEW_TOKENS = 200
DEVICE = "cpu"

PROBLEMS = [
    "Problem: What is the sum of the first 10 natural numbers?\n\nSolution:",
    "Problem: If a triangle has sides of length 3, 4, and 5, what is its area?\n\nSolution:",
    "Problem: Solve for x: 2x + 6 = 14\n\nSolution:",
    "Problem: What is 15% of 200?\n\nSolution:",
    "Problem: A car travels 60 miles per hour. How far does it travel in 2.5 hours?\n\nSolution:",
]

def generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("=" * 60)
    print("  GEKO GPT-2 Inference Test")
    print("=" * 60)

    print("\nLoading base GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    base_model.eval()
    print("  Base GPT-2 loaded.")

    print(f"\nLoading GEKO-finetuned checkpoint: {CHECKPOINT}...")
    geko_model = GPT2LMHeadModel.from_pretrained(CHECKPOINT).to(DEVICE)
    geko_model.eval()
    print("  GEKO model loaded.")

    print("\n" + "=" * 60)
    print("  Comparing Base vs GEKO-finetuned")
    print("=" * 60)

    for i, problem in enumerate(PROBLEMS, 1):
        print(f"\n{'─' * 60}")
        print(f"[Problem {i}]")
        # Print just the problem part
        print(problem.split("\n\nSolution:")[0].replace("Problem: ", ""))
        print()

        base_ans  = generate(base_model,  tokenizer, problem)
        geko_ans  = generate(geko_model,  tokenizer, problem)

        print(f"  BASE GPT-2 : {base_ans[:300]}")
        print()
        print(f"  GEKO GPT-2 : {geko_ans[:300]}")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
