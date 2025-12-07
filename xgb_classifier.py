import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score  
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from helpers import *

# 1. Load features data
df = pd.read_csv("atm_dataset.csv")
X = df.filter(regex="^atm_").values
y = df["label"].values

print("X shape:", X.shape)
print("y distribution BEFORE split:", np.bincount(y))

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Class distribution in TRAIN:", np.bincount(y_train))
print("Class distribution in TEST:",  np.bincount(y_test))

# 3. Pipeline with XGBoost
pipeline = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy={1:14}, random_state=42)),
    ('smote', SMOTE(sampling_strategy={0:14}, random_state=42)),
    ('scaler', StandardScaler()),  
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        tree_method='hist'   # FAST, good for high dimensional data
    ))
])

# 4. Hyperparameter grid search
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

params = {
    # Tree complexity
    'xgb__max_depth': [3, 5, 7],
    'xgb__min_child_weight': [1, 5, 10],

    # Learning rate
    'xgb__learning_rate': [0.01, 0.05, 0.1],

    # Number of trees
    'xgb__n_estimators': [100, 300, 500],

    # Regularization 
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest XGBoost parameters:", grid.best_params_)
print("Best CV balanced accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# 5. Evaluate on test set
y_pred = best_model.predict(X_test)
test_bal_acc = balanced_accuracy_score(y_test, y_pred)
print("\nTest balanced accuracy:", test_bal_acc)
print(classification_report(y_test, y_pred))

# 6. Save results
save_results_to_excel(
    model_name="XGBoost",
    best_params=grid.best_params_,
    cv_score=grid.best_score_,
    test_score=test_bal_acc,
    filename="results.xlsx"
)