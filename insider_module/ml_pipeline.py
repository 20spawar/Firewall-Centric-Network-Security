"""
Insider Threat Detection - ML Pipeline
Uses CERT r4.2 sampled data to train Isolation Forest + SHAP explanations
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# Paths
SAMPLES_DIR = r"D:\MAJOR PROJECT DATA\CERT R4.2\r4.2\r4.2\samples"
OUTPUT_DIR = r"D:\MAJOR PROJECT DATA\CERT R4.2\r4.2\r4.2\ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("INSIDER THREAT DETECTION - ML PIPELINE")
print("=" * 70)

# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1/6] Loading sampled data...")
insiders_df = pd.read_csv(os.path.join(SAMPLES_DIR, "insiders_sample.csv"))
logon_df = pd.read_csv(os.path.join(SAMPLES_DIR, "logon_sample.csv"), parse_dates=['date'])
device_df = pd.read_csv(os.path.join(SAMPLES_DIR, "device_sample.csv"), parse_dates=['date'])
file_df = pd.read_csv(os.path.join(SAMPLES_DIR, "file_sample.csv"), parse_dates=['date'])
email_df = pd.read_csv(os.path.join(SAMPLES_DIR, "email_sample.csv"), parse_dates=['date'])

print(f"✓ Insiders: {len(insiders_df)} users")
print(f"✓ Logon records: {len(logon_df)}")
print(f"✓ Device records: {len(device_df)}")
print(f"✓ File records: {len(file_df)}")
print(f"✓ Email records: {len(email_df)}")

# Get malicious users
malicious_users = set(insiders_df['user'].unique())
print(f"✓ Malicious users: {len(malicious_users)}")

# ==================== STEP 2: FEATURE ENGINEERING ====================
print("\n[STEP 2/6] Feature Engineering...")

# Get unique users
all_users = set(logon_df['user'].unique()) | set(device_df['user'].unique()) | set(file_df['user'].unique()) | set(email_df['user'].unique())
print(f"✓ Total unique users: {len(all_users)}")

features_list = []

for user in sorted(all_users):
    user_features = {'user': user}
    
    # LOGON FEATURES
    user_logon = logon_df[logon_df['user'] == user]
    if len(user_logon) > 0:
        user_features['logon_count'] = len(user_logon)
        user_features['logon_unique_hours'] = user_logon['date'].dt.hour.nunique()
        user_features['logon_after_hours'] = len(user_logon[(user_logon['date'].dt.hour < 7) | (user_logon['date'].dt.hour > 18)])
        user_features['logon_weekend'] = len(user_logon[user_logon['date'].dt.weekday >= 5])
    else:
        user_features['logon_count'] = 0
        user_features['logon_unique_hours'] = 0
        user_features['logon_after_hours'] = 0
        user_features['logon_weekend'] = 0
    
    # DEVICE FEATURES
    user_device = device_df[device_df['user'] == user]
    if len(user_device) > 0:
        user_features['device_connect_count'] = len(user_device)
        user_features['usb_connects'] = len(user_device[user_device['activity'] == 'Connect'])
        user_features['usb_disconnects'] = len(user_device[user_device['activity'] == 'Disconnect'])
    else:
        user_features['device_connect_count'] = 0
        user_features['usb_connects'] = 0
        user_features['usb_disconnects'] = 0
    
    # FILE FEATURES
    user_file = file_df[file_df['user'] == user]
    if len(user_file) > 0:
        user_features['file_operations_count'] = len(user_file)
        user_features['unique_files_accessed'] = user_file['filename'].nunique()
        # Count file operations by common content types
        user_features['file_exe_access'] = len(user_file[user_file['filename'].str.contains('.exe', case=False, na=False)])
        user_features['file_doc_access'] = len(user_file[user_file['filename'].str.contains(r'\.(doc|docx|pdf|txt)', case=False, na=False)])
    else:
        user_features['file_operations_count'] = 0
        user_features['unique_files_accessed'] = 0
        user_features['file_exe_access'] = 0
        user_features['file_doc_access'] = 0
    
    # EMAIL FEATURES
    user_email = email_df[email_df['user'] == user]
    if len(user_email) > 0:
        user_features['total_emails'] = len(user_email)
        user_features['emails_with_attachment'] = len(user_email[user_email['attachments'] > 0])
        user_features['unique_recipients'] = user_email['to'].nunique()
        user_features['avg_email_size'] = user_email['size'].mean()
    else:
        user_features['total_emails'] = 0
        user_features['emails_with_attachment'] = 0
        user_features['unique_recipients'] = 0
        user_features['avg_email_size'] = 0
    
    # LABEL
    user_features['is_malicious'] = 1 if user in malicious_users else 0
    
    features_list.append(user_features)

features_df = pd.DataFrame(features_list)
print(f"✓ Created {len(features_df.columns)} features for {len(features_df)} users")

# Save features
features_df.to_csv(os.path.join(OUTPUT_DIR, "user_features.csv"), index=False)
print(f"✓ Saved: user_features.csv")

# ==================== STEP 3: DATA PREPARATION ====================
print("\n[STEP 3/6] Preparing data for ML...")

# Select feature columns (exclude user and label)
feature_cols = [col for col in features_df.columns if col not in ['user', 'is_malicious']]
X = features_df[feature_cols].fillna(0)
y = features_df['is_malicious']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Samples: {X.shape[0]}")
print(f"✓ Malicious users: {y.sum()}")
print(f"✓ Normal users: {(1-y).sum()}")

# ==================== STEP 4: TRAIN ISOLATION FOREST ====================
print("\n[STEP 4/6] Training Isolation Forest...")

iso_forest = IsolationForest(contamination=len(malicious_users)/len(features_df), random_state=42, n_estimators=100)
anomaly_pred = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

# Convert anomaly scores to 0-100 risk score
risk_score = (1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())) * 100

features_df['anomaly_score'] = anomaly_scores
features_df['risk_score'] = risk_score
features_df['predicted_anomaly'] = (anomaly_pred == -1).astype(int)

print(f"✓ Model trained")
print(f"✓ Predicted anomalies: {(anomaly_pred == -1).sum()}")

# ==================== STEP 5: EVALUATION ====================
print("\n[STEP 5/6] Evaluating model...")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

y_pred = features_df['predicted_anomaly']
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Normal', 'Anomaly']))

roc_auc = roc_auc_score(y, risk_score)
print(f"\nROC-AUC Score: {roc_auc:.3f}")

# ==================== STEP 6: SHAP EXPLANATIONS ====================
print("\n[STEP 6/6] Generating SHAP explanations...")

# Use smaller sample for SHAP (computational efficiency)
sample_size = min(100, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X_sample)

print(f"✓ SHAP values computed for {len(X_sample)} samples")

# ==================== RESULTS ====================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE ✓")
print("=" * 70)

# Top malicious users by risk score
print("\n🔴 TOP 10 HIGHEST RISK USERS:")
top_risk = features_df.nlargest(10, 'risk_score')[['user', 'risk_score', 'is_malicious', 'file_operations_count', 'usb_connects', 'total_emails']]
print(top_risk.to_string(index=False))

# Correctly identified malicious users
correctly_identified = features_df[(features_df['is_malicious'] == 1) & (features_df['predicted_anomaly'] == 1)]
print(f"\n✓ Correctly identified malicious users: {len(correctly_identified)}/{len(malicious_users)} ({len(correctly_identified)/len(malicious_users)*100:.1f}%)")

# Save results
features_df.to_csv(os.path.join(OUTPUT_DIR, "user_predictions.csv"), index=False)
print(f"\n✓ Saved: user_predictions.csv")

# ==================== FUSION OUTPUT ====================
print("\nPreparing Insider Module output for Fusion...")

insider_output = features_df[
    ['user', 'risk_score', 'predicted_anomaly']
].copy()

insider_output.rename(
    columns={
        'risk_score': 'insider_score',
        'predicted_anomaly': 'is_insider'
    },
    inplace=True
)

INSIDER_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "insider_scores.csv"
)

os.makedirs(os.path.dirname(INSIDER_OUTPUT_PATH), exist_ok=True)

insider_output.to_csv(
    INSIDER_OUTPUT_PATH,
    index=False
)

print(f"✓ Insider Fusion output saved to: {INSIDER_OUTPUT_PATH}")
print("\nInsider output preview:")
print(insider_output.head())

# Create visualizations
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Risk score distribution
axes[0, 0].hist(features_df[features_df['is_malicious'] == 0]['risk_score'], bins=30, alpha=0.7, label='Normal', color='green')
axes[0, 0].hist(features_df[features_df['is_malicious'] == 1]['risk_score'], bins=30, alpha=0.7, label='Malicious', color='red')
axes[0, 0].set_xlabel('Risk Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Risk Score Distribution')
axes[0, 0].legend()

# Feature importance (from SHAP mean)
mean_shap = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(mean_shap)[-10:]
axes[0, 1].barh(range(len(top_features)), mean_shap[top_features])
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels([feature_cols[i] for i in top_features])
axes[0, 1].set_xlabel('Mean |SHAP value|')
axes[0, 1].set_title('Top 10 Important Features')

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xticklabels(['Normal', 'Anomaly'])
axes[1, 0].set_yticklabels(['Normal', 'Anomaly'])

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, risk_score)
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_evaluation.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved: model_evaluation.png")

print("\n" + "=" * 70)
print(f"All results saved to: {OUTPUT_DIR}")
print("=" * 70)
print("\nFiles created:")
print("  • user_features.csv - All extracted features")
print("  • user_predictions.csv - Final predictions + risk scores")
print("  • model_evaluation.png - Visualizations")
print("\n✓ INSIDER THREAT DETECTION COMPLETE!")