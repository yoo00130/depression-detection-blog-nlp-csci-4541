"""
Preprocess the Mental Health Blog Dataset for
binary depression detection.

Input:
  mental_health_blog_dataset.csv

Output:
  data/mhb_train.jsonl
  data/mhb_dev.jsonl
  data/mhb_test.jsonl

Label definition:
  1 = depression
  0 = non-depression (anxiety, ptsd-trauma, suicidal-thoughts-and-self-harm)
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to mental_health_blog_dataset.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed JSONL files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for train/dev/test split",
    )
    parser.add_argument(
        "--only_original_posts",
        action="store_true",
        help="If set, keep only posts where is_reply == 0",
    )
    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(args.input_csv)
    original_size = len(df)

    # -----------------------------
    # Remove empty / missing text
    # -----------------------------
    df = df[df["post"].notna()].copy()
    df["post"] = df["post"].astype(str).str.strip()
    df = df[df["post"].str.len() > 0].copy()

    cleaned_size = len(df)

    # -----------------------------
    # Create binary label
    # -----------------------------
    df["label"] = (df["category"] == "depression").astype(int)

    # Create a stable ID
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)

    # Final dataset
    data = df[["id", "post", "label"]].rename(columns={"post": "text"})

    # -----------------------------
    # Train / Dev / Test split
    # -----------------------------
    train_df, temp_df = train_test_split(
        data,
        test_size=0.2,
        random_state=args.seed,
        stratify=data["label"],
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    # -----------------------------
    # Save files
    # -----------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "mhb_train.jsonl")
    dev_path = os.path.join(args.output_dir, "mhb_dev.jsonl")
    test_path = os.path.join(args.output_dir, "mhb_test.jsonl")

    train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)
    dev_df.to_json(dev_path, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_path, orient="records", lines=True, force_ascii=False)

    # -----------------------------
    # Sanity checks / logging
    # -----------------------------
    print("Preprocessing complete.\n")
    print(f"Original rows: {original_size}")
    print(f"Rows after cleaning: {cleaned_size} "
          f"(dropped {original_size - cleaned_size})\n")

    print(f"Train size: {len(train_df)}")
    print(f"Dev size:   {len(dev_df)}")
    print(f"Test size:  {len(test_df)}\n")

    print("Label distribution (train):")
    print(train_df["label"].value_counts(normalize=True))
    print("\nSaved files:")
    print(" ", train_path)
    print(" ", dev_path)
    print(" ", test_path)


if __name__ == "__main__":
    main()
