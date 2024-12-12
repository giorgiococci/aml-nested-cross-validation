import argparse
import mlflow
import os

import pandas as pd

from pathlib import Path
from sklearn.svm import SVC

def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = SVC(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model

def main(args):
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    mlflow.autolog()
    
    # setup parameters
    params = {
        "C": args.C,
        "kernel": args.kernel,
        "degree": args.degree,
        "gamma": args.gamma,
        "coef0": args.coef0,
        "shrinking": args.shrinking,
        "probability": args.probability,
        "tol": args.tol,
        "cache_size": args.cache_size,
        "class_weight": args.class_weight,
        "verbose": args.verbose,
        "max_iter": args.max_iter,
        "decision_function_shape": args.decision_function_shape,
        "break_ties": args.break_ties,
        "random_state": args.random_state,
    }
    
    datasets_path = args.datasets_path
    
    # In the datasets_path there are two files: "*_train.csv" and "*_test.csv"
    # Get training and testing files
    train_files = [f for f in os.listdir(datasets_path) if f.endswith("_train.csv")]
    test_files = [f for f in os.listdir(datasets_path) if f.endswith("_test.csv")]
    
    # Load the training and testing datasets
    train_datasets = [pd.read_csv(os.path.join(datasets_path, f)) for f in train_files]
    test_datasets = [pd.read_csv(os.path.join(datasets_path, f)) for f in test_files]
    
    # Get only the first train and test datasets
    train_dataset = train_datasets[0]
    test_dataset = test_datasets[0]
    
    # process data
    X_train, y_train = train_dataset.drop(["species"], axis=1), train_dataset["species"]
    X_test, y_test = test_dataset.drop(["species"], axis=1), test_dataset["species"]
    
    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)
    # Output the model and test data
    # write to local folder first, then copy to output folder

    mlflow.sklearn.save_model(model, "model")

    from distutils.dir_util import copy_tree

    # copy subdirectory example
    from_directory = "model"
    to_directory = args.model_output

    copy_tree(from_directory, to_directory)

    X_test.to_csv(Path(args.test_data) / "X_test.csv", index=False)
    y_test.to_csv(Path(args.test_data) / "y_test.csv", index=False)
    

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--datasets_path", type=str, required=True, help="Path to the dataset CSV files")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--coef0", type=float, default=0)
    parser.add_argument("--shrinking", type=bool, default=False)
    parser.add_argument("--probability", type=bool, default=False)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache_size", type=float, default=1024)
    parser.add_argument("--class_weight", type=dict, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--decision_function_shape", type=str, default="ovr")
    parser.add_argument("--break_ties", type=bool, default=False)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_data", type=str, help="Path of output model")

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # run main
    main(args)