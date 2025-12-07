import numpy as np
from helpers import *
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline

# 1. Load features data
df = pd.read_csv("coherence_dataset.csv")
X = df.filter(regex="^coh_").values
y = df["label"].values
print("X shape:", X.shape)
print("y distribution:", np.bincount(y))

# 2. Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Class distribution in TRAIN:", np.bincount(y_train))
print("Class distribution in TEST:",  np.bincount(y_test))

# 3. Pipeline 
pipeline = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy={1:20}, random_state=42)), 
    ('smote', SMOTE(sampling_strategy={0:20}, random_state=42)),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(metric="euclidean"))
])

# 4. Grid search for best k
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
params = {
    'knn__n_neighbors': [1, 3, 5, 7],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
    }
grid = GridSearchCV(pipeline, params, cv=cv, scoring='balanced_accuracy') 
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

# 5. Evaluate
y_pred = best_model.predict(X_test)

print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Best K:", grid.best_params_['knn__n_neighbors'])
print("Best weights:", grid.best_params_['knn__weights'])
print("Best metric:", grid.best_params_['knn__metric'])


# PIPELINE WITH PCA (optional)

# 3b. Pipeline with PCA:
pipeline_PCA = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy={1:20}, random_state=42)), 
    ('smote', SMOTE(sampling_strategy={0:20}, random_state=42)),
    ('pca', PCA(n_components=0.95, svd_solver='full')), 
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(metric="euclidean"))
])

# 4b. Grid search for best k with PCA

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
params_PCA = {
    'knn__n_neighbors': [1, 3, 5, 7],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],    
    }
grid = GridSearchCV(pipeline_PCA, params_PCA, cv=cv, scoring='balanced_accuracy') #cv already does cross validation when doing grid search -> change only if stratified is desired
grid.fit(X_train, y_train)

print("Best pca components and k:", grid.best_params_)
print("Best CV score with PCA:", grid.best_score_)

best_model = grid.best_estimator_

# 5b. Evaluate with PCA
y_pred = best_model.predict(X_test)

print("Balanced accuracy with PCA:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Best K:", grid.best_params_['knn__n_neighbors'])
print("Best weights:", grid.best_params_['knn__weights'])
print("Best metric:", grid.best_params_['knn__metric'])