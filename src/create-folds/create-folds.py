import json
import argparse
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd

def create_outer_folds(dataset_path, n_splits, output_path, random_state=None):
    """
    Create train-test splits for outer cross-validation folds.

    Args:
        dataset_path (str): Path to the dataset CSV file.
        n_splits (int): Number of outer folds.
        output_path (str): Path to save the JSON file with fold indices.
        random_state (int, optional): Random state for reproducibility.
    """
    # Load dataset
    data = pd.read_csv(dataset_path)
    n_samples = data.shape[0]

    # Perform outer cross-validation splitting
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_index, test_index in kf.split(range(n_samples)):
        folds.append({"train": train_index.tolist(), "test": test_index.tolist()})
        
    # Save each fold to a separate JSON file
    for i, fold in enumerate(folds):
        with open(Path(output_path) / f"fold_{i}.json", "w") as f:
            json.dump(fold, f, indent=4)

    print(f"Saved outer fold splits to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of outer folds")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the fold splits JSON file")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    create_outer_folds(args.dataset_path, args.n_splits, args.output_path, args.random_state)
