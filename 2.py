import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

print("Starting Machine Learning Project...")

# ==========================================
# PHASE 1: GENERATE SYNTHETIC DATA
# ==========================================
print("\n[1/4] Generating movie dataset...")
np.random.seed(42)

n_samples = 500
# Features: Budget (in Crores), Actor Tier (1=Top, 3=Low), Director Win Rate (%), Pre-release Hype (1-10)
budgets = np.random.uniform(10, 300, n_samples)
actor_tiers = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.5, 0.3])
director_win_rates = np.random.uniform(20, 90, n_samples)
hype_index = np.random.uniform(1, 10, n_samples)

# Target Variable: Is_Hit (1 = Hit, 0 = Flop)
# Logic: Higher budget + top actor + good director + high hype = higher chance of being a hit
success_probability = (
    (budgets / 300) * 0.2 + 
    ((4 - actor_tiers) / 3) * 0.3 + 
    (director_win_rates / 100) * 0.3 + 
    (hype_index / 10) * 0.2
)

# Add some randomness/noise so the model has to learn
is_hit = np.where(success_probability + np.random.normal(0, 0.1, n_samples) > 0.55, 1, 0)

df = pd.DataFrame({
    'Budget_Cr': budgets,
    'Actor_Tier': actor_tiers,
    'Director_Win_Rate': director_win_rates,
    'Hype_Index': hype_index,
    'Is_Hit': is_hit
})

print("--> Dataset created. First 3 rows:")
print(df.head(3))

# ==========================================
# PHASE 2: TRAIN-TEST SPLIT
# ==========================================
print("\n[2/4] Splitting data into training and testing sets...")
X = df.drop('Is_Hit', axis=1) # Features
y = df['Is_Hit']              # Target

# Split: 80% for training the model, 20% for testing its predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--> Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ==========================================
# PHASE 3: MODEL TRAINING & PREDICTION
# ==========================================
print("\n[3/4] Training Random Forest Classifier...")
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

# Calculate basic accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"--> Model Accuracy: {accuracy * 100:.2f}%\n")
print("--> Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# PHASE 4: VISUALIZING PERFORMANCE
# ==========================================
print("\n[4/4] Generating evaluation charts...")
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Visualization 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Predicted Flop', 'Predicted Hit'],
            yticklabels=['Actual Flop', 'Actual Hit'])
axes[0].set_title("Confusion Matrix")

# Visualization 2: ROC Curve
# The ROC curve plots the True Positive Rate against the False Positive Rate
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC)')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

print("\nProject execution complete!")