#analyze the distribution of SHAP values across all the seeds

import sklearn
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


# Creating useful folders for saving outputs
plot_output_dir = Path("shap_plots")
plot_output_dir.mkdir(exist_ok=True)
shap_output_dir = Path("shap_values")
shap_output_dir.mkdir(exist_ok=True)

# 1. Load data
df_atm = pd.read_csv("atm_dataset.csv")
X_atm = df_atm.filter(regex="^atm_").values
y_atm = df_atm["label"].values

sample_size = min(100, X_atm.shape[0])
random_indices = np.random.choice(X_atm.shape[0], size=sample_size, replace=False)
X_sample = X_atm[random_indices]

# 2. Load models and compute SHAP values across seeds
output_dir = Path("xgb_best_models")
feature_names = df_atm.filter(regex="^atm_").columns
all_shap_values = []
shap_objects = []

for seed in range(51):
    model_path = output_dir / f"best_xgb_model_seed_{seed}.pkl"
    
    if not model_path.exists():
        print(f"Model for seed {seed} not found, skipping...")
        continue
    
    xgb_best = joblib.load(model_path)
    
    # Scale data
    scaler = xgb_best.named_steps["scaler"]
    X_scaled = scaler.transform(X_sample)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb_best.named_steps["xgb"])
    shap_full = explainer(X_scaled)
    shap_values = shap_full.values
    
    all_shap_values.append(shap_values)
    shap_objects.append(shap_full)
    print(f"Processed seed {seed}")

all_shap_values = np.array(all_shap_values)  # Shape: (50, sample_size, n_features)

# 3. Plot SHAP distribution across seeds
# Calculate mean absolute SHAP values per feature across all seeds and samples
mean_shap_per_feature = np.abs(all_shap_values).mean(axis=(0, 1))
std_shap_per_feature = np.abs(all_shap_values).mean(axis=1).std(axis=0)

# Feature importance bar plot with error bars
fig, ax = plt.subplots(figsize=(12, 6))
features = list(feature_names)
indices = np.argsort(mean_shap_per_feature)[::-1]

ax.barh(range(len(indices)), mean_shap_per_feature[indices], 
        xerr=std_shap_per_feature[indices], capsize=5, alpha=0.7)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices])
ax.set_xlabel("Mean |SHAP value| (across 50 seeds)")
ax.set_title("SHAP Feature Importance Distribution Across 50 Seeds")
plt.tight_layout()
plt.savefig(plot_output_dir / "shap_distribution_across_seeds.png", dpi=300)
plt.show()

# 4a. Plot beeswarm plot for each seed
for seed, shap_obj in enumerate(shap_objects):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_obj, show=False)
    plt.title(f"SHAP Beeswarm Plot - Seed {seed}")
    plt.tight_layout()
    plt.savefig(plot_output_dir / f"shap_beeswarm_seed_{seed}.png", dpi=300)
    plt.close()
    print(f"Saved beeswarm plot for seed {seed}")

# 4b. Create aggregated beeswarm plot across all seeds
# Average SHAP values across seeds
mean_shap_values = np.mean(all_shap_values, axis=0)
std_shap_values = np.std(all_shap_values, axis=0)

# Create a mock SHAP object with aggregated values, shaped (N_SEEDS, n_test_samples, n_features)
shap_3d_array = np.stack(all_shap_values, axis=0)

# Mean SHAP values over all 50 seeds (axis = 0), resulting shape: (n_test_samples, n_features)
mean_shap_values = np.mean(shap_3d_array, axis=0)

# Calculating mean absolute feature importance across all seeds and all samples
    # 1. Take absolute values
    # 2. Mean over all seeds (axis=0)
    # Resulting shape: (n_features, )
mean_abs_feature_importance = np.mean(np.abs(mean_shap_values), axis=0)
print("-" * 30)
print(f"Mean SHAP Value over 50 seeds shape: {mean_shap_values.shape}")
print(f"Shape of mean absolute feature importance (over 50 seeds, per feature): {mean_abs_feature_importance.shape}")
fig, ax = plt.subplots(figsize=(12, 6))
# Flatten all SHAP values across all seeds to show individual sample predictions
all_shap_flat = all_shap_values.reshape(-1, all_shap_values.shape[-1])
# Beeswarm plot for the aggregated SHAP values
shap.plots.beeswarm(shap.Explanation(values=all_shap_flat, feature_names=list(feature_names)), show=False)
plt.title("SHAP Beeswarm Plot - Aggregated Across All 50 Seeds")
plt.tight_layout()
plt.savefig(plot_output_dir / "shap_beeswarm_aggregated_all_seeds.png", dpi=300)
plt.show()
print("Saved aggregated beeswarm plot across all seeds")
# Bar plot for the aggregated SHAP values
shap.plots.bar(shap.Explanation(values=all_shap_flat, feature_names=list(feature_names)), show=False)
plt.title("SHAP Bar Plot - Aggregated Across All 50 Seeds")
plt.tight_layout()
plt.savefig(plot_output_dir / "shap_bar_aggregated_all_seeds.png", dpi=300)
plt.show()
print("Saved aggregated bar plot across all seeds")
print("SHAP analysis completed.")


# 5. Reconstruct SHAP values to ATM regions for each seed
region_names = [f"ROI_{i+1}" for i in range(62)]
n_regions = len(region_names)

all_roi_dfs = []

for seed_idx, shap_values in enumerate(all_shap_values):
    # shap_values shape: (sample_size, n_features)
    # Compute mean absolute SHAP values across samples
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    shap_values_atm = shap_mean.reshape(62, 62)
    
    # Map SHAP values back to atlas regions
    records = []
    for i in range(n_regions):
        for j in range(n_regions):
            records.append({
                "seed": seed_idx,
                "ROI_i": region_names[i],
                "ROI_j": region_names[j],
                "mean_abs_shap": shap_values_atm[i, j]
            })
    #one data frame per seed
    df = pd.DataFrame(records)
    df = df.sort_values("mean_abs_shap", ascending=False) # sorting the values in descending order, from the most influential regions to the least
    df.to_csv(shap_output_dir / f"shap_values_regions_seed{seed_idx}.csv", index=False)
    all_roi_dfs.append(df)
    print(f"SHAP values mapped to ATM regions and saved to shap_values_regions_seed{seed_idx}.csv")
    
# Combine all seeds
shap_roi_df = pd.concat(all_roi_dfs, ignore_index=True)

# Save to CSV
shap_roi_df.to_csv(shap_output_dir / "shap_values_atm_regions_all_seeds.csv", index=False)
# Save to Excel with separate sheets for each seed, and an introductory sheet with summary statistics for the most influential regions overall
with pd.ExcelWriter("shap_values_atm_regions_all_seeds.xlsx", engine="openpyxl") as writer:
    # Summary sheet
    summary_df = shap_roi_df.groupby("seed")["mean_abs_shap"].describe().round(6)
    summary_df.to_excel(writer, sheet_name="Summary")
    
    # Individual sheets for each seed showing the most influential ROI pairs for each seed 
    for seed_idx, seed_df in enumerate(all_roi_dfs):
        seed_df.to_excel(writer, sheet_name=f"Seed_{seed_idx}", index=False)
    
    # Interpretation sheet: aggregate across all seeds
    interpretation_records = []
    for i in range(n_regions):
        for j in range(n_regions):
            # Get mean and std SHAP value across all seeds for this ROI pair
            roi_pair_data = shap_roi_df[
                (shap_roi_df["ROI_i"] == region_names[i]) & 
                (shap_roi_df["ROI_j"] == region_names[j])
            ]["mean_abs_shap"]
            interpretation_records.append({
                "ROI_i": region_names[i],
                "ROI_j": region_names[j],
                "mean_abs_shap": roi_pair_data.mean(),
                "std_abs_shap": roi_pair_data.std(),
            })
    
    interpretation_df = pd.DataFrame(interpretation_records)
    interpretation_df = interpretation_df.sort_values("mean_abs_shap", ascending=False).round(6)
    interpretation_df.to_excel(writer, sheet_name="Interpretation", index=False)

    # DataFrame for singular node importance
    roi_importance_records = []
    
    for region in region_names:
        
        # Extract each SHAP value where the region is present, using aggregated DataFrame over all seeds
        data_i = interpretation_df[interpretation_df["ROI_i"] == region]["mean_abs_shap"]
        data_j = interpretation_df[interpretation_df["ROI_j"] == region]["mean_abs_shap"]
        
        # Combine SHAP values where the region is either ROI_i or ROI_j
        all_shap_values_for_roi = pd.concat([data_i, data_j])
        
        # Mean and standard deviation of SHAP values for this region
        mean_node_importance = all_shap_values_for_roi.mean()
        std_node_importance = all_shap_values_for_roi.std()
        
        
        roi_importance_records.append({
            "ROI": region,
            "mean_node_importance": mean_node_importance,
            "std_node_importance": std_node_importance
        })
    
    node_importance_df = pd.DataFrame(roi_importance_records)
    node_importance_df = node_importance_df.sort_values("mean_node_importance", ascending=False).round(6)
    
    node_importance_df.to_excel(writer, sheet_name="Node_Importance", index=False)
    
    # Reorder sheets to put Node_Importance first
    writer.book.move_sheet("Interpretation", offset=-writer.book.index(writer.book["Interpretation"]))

print("SHAP values mapped to ATM regions across all seeds")
print(shap_roi_df.groupby("seed")["mean_abs_shap"].describe())

# 6. Plot aggregated ATM SHAP values heatmap
aggregated_shap_atm = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        aggregated_shap_atm[i, j] = shap_roi_df[
            (shap_roi_df["ROI_i"] == region_names[i]) & 
            (shap_roi_df["ROI_j"] == region_names[j])
        ]["mean_abs_shap"].mean()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(aggregated_shap_atm, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(n_regions))
ax.set_yticks(range(n_regions))
ax.set_xticklabels(region_names, rotation=90, fontsize=8)
ax.set_yticklabels(region_names, fontsize=8)
ax.set_xlabel("ROI")
ax.set_ylabel("ROI")
ax.set_title("Aggregated SHAP Values for ATM Connections (Across 50 Seeds)")
plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
plt.tight_layout()
plt.savefig(plot_output_dir / "shap_atm_heatmap_aggregated.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved aggregated ATM SHAP heatmap")
