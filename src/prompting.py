"""
Zero-shot prompting baseline using facebook/bart-large-mnli.

This uses an NLI-based zero-shot classifier (no training).
Evaluation is performed on a random subset of the test set
to control runtime.
"""

import argparse
import json
import os
import random

import numpy as np
from datasets import load_dataset
from transformers import pipeline

from src.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # -----------------------------
    # Load test data
    # -----------------------------
    dataset = load_dataset("json", data_files={"test": args.test_file})
    texts = dataset["test"]["text"]
    gold = dataset["test"]["label"]

    # -----------------------------
    # Subsample test set
    # -----------------------------
    random.seed(args.seed)
    indices = random.sample(range(len(texts)), min(args.sample_size, len(texts)))

    sampled_texts = [texts[i] for i in indices]
    sampled_gold = [gold[i] for i in indices]

    print(f"Running zero-shot prompting with BART-large-MNLI on {len(sampled_texts)} samples")

    # -----------------------------
    # Zero-shot classifier
    # -----------------------------
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU only
    )

    candidate_labels = ["depression", "non-depression"]

    preds = []
    for i, text in enumerate(sampled_texts):
        result = classifier(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template="This text expresses {}.",
        )

        # Pick the label with the highest score
        pred_label = result["labels"][0]
        pred = 1 if pred_label == "depression" else 0
        preds.append(pred)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} examples")

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = compute_metrics(sampled_gold, preds)

    print("\nZero-shot prompting results (BART-large-MNLI):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    out_path = "outputs/prompting_bart_mnli_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "facebook/bart-large-mnli",
                "method": "zero-shot",
                "sample_size": len(sampled_texts),
                "seed": args.seed,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print("\nSaved metrics to:")
    print(" ", out_path)


if __name__ == "__main__":
    main()
