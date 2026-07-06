import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from openai import OpenAI

from meme_stock_classifier import classify_dataframe, FAILURE_LABEL

INPUT_FILE = "wsb_test_holdout_v4.csv"
load_dotenv()

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not found in .env file!\n"
            "Please create a .env file in the project root with your key.")

    print("✅ API key successfully loaded from .env")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_FILE)

    if "true_label" not in df.columns:
        raise SystemExit(f"No column [true_label] in file \'{INPUT_FILE}\'. First you need labeling.")

    df = df.dropna(subset=["true_label"])
    df = df[df["true_label"] != ""]
    if df.empty:
        raise SystemExit("Column [true_label] is empty — nothing to evaluate.")

    print(f"Classifying {len(df)} posts using the updated classifier...")
    result = classify_dataframe(df, client=client)

    valid = result[result["sentiment"] != FAILURE_LABEL]
    failed_count = len(result) - len(valid)

    acc = accuracy_score(valid["true_label"], valid["sentiment"])
    print(f"\nAccuracy: {acc:.1%}  (on {len(valid)} out of {len(result)} posts, "
          f"{failed_count} failed to classify)")

    print("\n--- Classification Report ---")
    print(classification_report(valid["true_label"], valid["sentiment"]))

    print("\n--- Confusion Matrix ---")
    labels = ["Positive", "Negative", "Neutral"]
    cm = confusion_matrix(valid["true_label"], valid["sentiment"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)

    print("\n--- Classification Method Breakdown ---")
    print(result["classification_method"].value_counts())

    result.to_csv("eval_results.csv", index=False)
    print("\nFull results saved to eval_results.csv")


if __name__ == "__main__":
    main()
