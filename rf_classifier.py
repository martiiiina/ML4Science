import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score  
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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

# 3. Pipeline with Random Forest
pipeline = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy={1:20})),
    ('smote', SMOTE(sampling_strategy={0:20})),
    ('scaler', StandardScaler()),   # optional for RF but okay to keep
    ('rf', RandomForestClassifier(random_state=42))
])

# 4. Grid search for RF parameters
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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

print("\nTest balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
