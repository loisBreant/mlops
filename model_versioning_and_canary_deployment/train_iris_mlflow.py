# !/usr/bin/env python3

import argparse
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on Iris and log with MLflow")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.25, help="Test set fraction")
    parser.add_argument("--run_name", type=str, default="iris-rf-run", help="MLflow run name")
    parser.add_argument("--save_model", action="store_true", help="Also log the trained model artifact (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Ensure tracking to local tp2/mlruns by default so Docker volume sees the runs
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        local_mlruns = os.path.abspath(os.path.join(os.path.dirname(__file__), "mlruns"))
        mlflow.set_tracking_uri(f"file://{local_mlruns}")
    # Load data
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train model
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # MLflow logging
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)

        mlflow.log_metric("accuracy", float(acc))

        if args.save_model:
            preds_train = clf.predict(X_train).astype("float64")
            signature = infer_signature(X_train, preds_train)
            input_example = X_train.head(5)
            # Log the model with a name (preferred API in modern MLflow)
            logged = mlflow.sklearn.log_model(
                sk_model=clf,
                name="random-forest-model",
                signature=signature,
                input_example=input_example,
            )

        print(f"Run ID: {run.info.run_id}")
        print(f"Logged accuracy: {acc:.4f}")
        print("Logged params:", {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "random_state": args.random_state})
        if args.save_model:
            try:
                print(f"MODEL_URI: {logged.model_uri}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
