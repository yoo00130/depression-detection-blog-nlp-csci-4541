"""
Fine-tune RoBERTa for binary depression detection (CPU-only, stable).

Input:
  data/mhb_train.jsonl
  data/mhb_dev.jsonl
  data/mhb_test.jsonl

Output:
  outputs/roberta/
    - test_metrics.json
    - test_predictions.csv
"""

import os
# IMPORTANT: disable MPS completely (Apple Silicon safety)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse
import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/roberta")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Running RoBERTa fine-tuning (CPU only)")

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.dev_file,
            "test": args.test_file,
        },
    )

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,   # reduced for memory safety
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.remove_columns(["text", "id"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    # -----------------------------
    # Model
    # -----------------------------
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    )

    # -----------------------------
    # Training arguments (CPU-only, no MPS, no eval tricks)
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        no_cuda=True,                    # 🔑 disables GPU/MPS
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        seed=args.seed,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # -----------------------------
    # Metrics
    # -----------------------------
    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_metrics(labels, preds)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=hf_compute_metrics,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Test evaluation
    # -----------------------------
    test_output = trainer.predict(dataset["test"])
    test_logits = test_output.predictions
    test_labels = test_output.label_ids
    test_preds = np.argmax(test_logits, axis=1)

    metrics = compute_metrics(test_labels, test_preds)

    print("\nTest set metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------
    # Save predictions
    # -----------------------------
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    with open(preds_path, "w") as f:
        f.write("gold,pred\n")
        for g, p in zip(test_labels, test_preds):
            f.write(f"{int(g)},{int(p)}\n")

    print("\nSaved outputs:")
    print(" ", metrics_path)
    print(" ", preds_path)


if __name__ == "__main__":
    main()
