import sklearn
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
#temporary fixing warning problem from xgboost, has to be resolves later

#STEPS:
# 1. Load data (from atm and coh datasets)
# 2. Load best models from knn_classifier.py and xgb_classifier.py
# 3. Compute SHAP values for both models
# 4. Plot SHAP summary plots



# 1. Load data
df_atm = pd.read_csv("atm_dataset.csv")
X_atm = df_atm.filter(regex="^atm_").values
y_atm = df_atm["label"].values

# 2. Load best models
#XGBoost or RandomForest: We’ll use one of these for model training. Both are popular in classification tasks, and SHAP integrates well with them.

from xgb_classifier import best_model # Assuming best_model is defined in xgb_classifier.py
xgb_best = best_model

sample_size = min(100, X_atm.shape[0])
random_indices = np.random.choice(X_atm.shape[0], size=sample_size, replace=False)
X_sample = X_atm[random_indices] #is it really that necessary to sample again here?

# 2b. Extract only the steps needed for inference
scaler = xgb_best.named_steps["scaler"]
model = xgb_best.named_steps["xgb"]     # the actual XGBoost model
#apply ONLY the scaler (not SMOTE, not undersample)
X_scaled = scaler.transform(X_sample)

#X_prep = xgb_best.named_steps["scaler"].transform(X_sample)

# 3. Compute SHAP values for model using TreeExplainer:
explainer = shap.TreeExplainer(model)
shap_full = explainer(X_scaled, check_additivity=False)
shap_values =shap_full.values

# Save SHAP values to a CSV file
shap_df = pd.DataFrame(shap_values, columns=df_atm.filter(regex="^atm_").columns)
shap_df.to_csv("shap_values_atm_xgb.csv", index=False)
print("SHAP values saved to shap_values_atm_xgb.csv")



# 4. Plot SHAP summary plots


#BEESWARM PLOT:
feature_names=df_atm.filter(regex="^atm_").columns
shap.summary_plot(
    shap_values, 
    X_sample, 
    feature_names=feature_names
    )

plt.show()

#PARTIAL DEPENDENCE:
#nderstand a feature’s importance in a model, it is necessary to understand both how changing that feature impacts the model’s output, and also the distribution of that feature’s values

feat_name = feature_names[0]  # e.g., the first ATM feature

shap.dependence_plot(
    feat_name,
    shap_values,   # or shap_values
    X_sample,
    feature_names=feature_names
)

# Get mean |SHAP| to rank features
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_k_idx = np.argsort(mean_abs_shap)[-10:][::-1]  # top 10

for idx in top_k_idx:
    shap.dependence_plot(
        feature_names[idx],
        shap_values,
        X_sample,
        feature_names=feature_names
    )


#WATERFALL PLOT:
#explain a single prediction
#shap.plots.waterfall(shap_values[0])

# Replace 0 with the index of the instance you want to explain
# choose an index in your sample
i = 0  # first sample in X_sample / X_scaled

# waterfall with the new API
shap.plots.waterfall(shap_full[i])
plt.show()

shap.waterfall_plot(
    shap_full.base_values[i],
    shap_full.values[i],
    feature_names=feature_names
)
plt.show()


print("SHAP analysis completed.")
