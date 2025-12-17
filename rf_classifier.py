import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score  
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from helpers import *




def run_single_seed(seed):
    print(f"\n===== Running SEED = {seed} =====")

    # 1. Load features data
    df = pd.read_csv("atm_dataset_30.csv")
    X = df.filter(regex="^atm_").values
    y = df["label"].values

    print("X shape:", X.shape)
    print("y distribution BEFORE split:", np.bincount(y))

    # 2. Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print("Class distribution in TRAIN:", np.bincount(y_train))
    print("Class distribution in TEST:",  np.bincount(y_test))

    # 3. Pipeline with Random Forest
    pipeline = Pipeline([
        ('undersample', RandomUnderSampler(sampling_strategy={1:14}, random_state=seed)),
        ('smote', SMOTE(sampling_strategy={0:14}, random_state=seed)),
        ('scaler', StandardScaler()),   # optional for RF but okay to keep
        ('rf', RandomForestClassifier(random_state=seed))
    ])

    # 4. Grid search for RF parameters
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    params = {
        'rf__n_estimators': [100, 300, 500],           # number of trees
        'rf__max_depth': [None, 10, 20, 40],           # tree depth
        'rf__min_samples_split': [2, 5, 10],           # when to split a node
        'rf__min_samples_leaf': [1, 2, 4],             # leaf size
        'rf__max_features': ['sqrt', 'log2'],          # feature subsampling
    }

    grid = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest RF parameters:", grid.best_params_)
    print("Best CV balanced accuracy:", grid.best_score_)

    best_model = grid.best_estimator_

    # 5. Evaluate
    y_pred = best_model.predict(X_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    print("\nTest balanced accuracy:", test_bal_acc)
    print(classification_report(y_test, y_pred))

    # 6. Save results
    save_results_to_excel(
        model_name="RF",
        best_params=grid.best_params_,
        cv_score=grid.best_score_,
        test_score=test_bal_acc,
        filename="results_atm.xlsx"
    )

    return {
        "seed": seed,
        "cv_score": grid.best_score_,
        "test_score": test_bal_acc,
        "best_params": grid.best_params_
    }

seeds = list(range(1, 51))

results = []

for seed in seeds:
    res = run_single_seed(seed)
    results.append(res)

# Convert in dataframes 
df_results = pd.DataFrame(results)
mean_test = df_results["test_score"].mean()
std_test = df_results["test_score"].std()
print("\n===== FINAL SUMMARY =====")
print(df_results)
print(f"\nMEAN Balanced Accuracy = {mean_test:.4f}")
print(f"STD Balanced Accuracy  = {std_test:.4f}")

