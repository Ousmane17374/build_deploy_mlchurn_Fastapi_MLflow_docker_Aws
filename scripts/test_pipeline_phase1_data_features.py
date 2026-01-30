# test_pipeline_phase1.py
import pandas as pd
import os

# Make sure python can find your src package
import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

#== Config==

DATA_PATH="D:/Applications/Build & Deploy ML churn model with FastAPI, MLFlow, Docker, & AWS/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL="Churn"
def main():
    print("===Testing phase 1: Load - Preprocess - Build Features ==")

    #1. Load Data
    print("\n[1] Loading data...")
    df=load_data(DATA_PATH)
    print(f"Data Loaded. shape: {df.shape}")
    print(df.head(3))

    #2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean=preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))