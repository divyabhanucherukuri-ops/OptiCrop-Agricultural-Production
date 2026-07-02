"""
OptiCrop - Model Training Script
Trains multiple classifiers and saves the best performing model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("OPTICROP - SMART AGRICULTURAL PRODUCTION OPTIMIZATION ENGINE")
print("=" * 60)
print("\n[1] Loading dataset...")
df = pd.read_csv('Crop_recommendation.csv')
print(f"    Dataset shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Crops: {df['label'].nunique()}")
print(f"    Samples per crop:\n{df['label'].value_counts()}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[2] Exploratory Data Analysis...")
print(f"\n    Dataset Info:")
print(f"    - Total samples: {len(df)}")
print(f"    - Features: {df.columns[:-1].tolist()}")
print(f"    - Target classes: {df['label'].nunique()} crop types")
print(f"\n    Statistical Summary:")
print(df.describe())

# Check for missing values
print(f"\n    Missing values:\n{df.isnull().sum()}")

# ============================================================
# 3. VISUALIZATION (saved to files)
# ============================================================
print("\n[3] Generating visualizations...")

# 3.1 Distribution of numerical features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4', '#E94F37']

for i, (feature, color) in enumerate(zip(features, colors)):
    row, col = i // 3, i % 3
    axes[row, col].hist(df[feature], bins=30, color=color, edgecolor='white', alpha=0.7)
    axes[row, col].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(alpha=0.3)

# Hide the last subplot (8th position, index 8)
if len(features) < 9:
    axes[2, 2].set_visible(False)

plt.tight_layout()
plt.savefig('static/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved feature_distributions.png")

# 3.2 Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(corr)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved correlation_heatmap.png")

# 3.3 Crop-wise Nutrient Requirements
crop_nutrients = df.groupby('label')[['N', 'P', 'K']].mean().sort_values('N', ascending=False)
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(crop_nutrients))
width = 0.25
ax.bar(x - width, crop_nutrients['N'], width, label='Nitrogen (N)', color='#2E86AB')
ax.bar(x, crop_nutrients['P'], width, label='Phosphorous (P)', color='#A23B72')
ax.bar(x + width, crop_nutrients['K'], width, label='Potassium (K)', color='#F18F01')
ax.set_xlabel('Crop Type', fontweight='bold')
ax.set_ylabel('Average Nutrient Level', fontweight='bold')
ax.set_title('Average NPK Requirements by Crop', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(crop_nutrients.index, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/crop_nutrient_requirements.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved crop_nutrient_requirements.png")

# 3.4 Environmental conditions by crop
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
env_features = ['temperature', 'humidity', 'ph', 'rainfall']
env_titles = ['Temperature Requirements (°C)', 'Humidity Requirements (%)', 
              'pH Requirements', 'Rainfall Requirements (mm)']

for i, (feat, title) in enumerate(zip(env_features, env_titles)):
    row, col = i // 2, i % 2
    crop_data = df.groupby('label')[feat].mean().sort_values()
    axes[row, col].barh(range(len(crop_data)), crop_data.values, color='#44BBA4', edgecolor='white')
    axes[row, col].set_yticks(range(len(crop_data)))
    axes[row, col].set_yticklabels(crop_data.index, fontsize=9)
    axes[row, col].set_title(title, fontsize=12, fontweight='bold')
    axes[row, col].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('static/environmental_conditions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved environmental_conditions.png")

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================
print("\n[4] Data preprocessing...")

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"    Classes: {label_encoder.classes_}")
print(f"    Number of classes: {len(label_encoder.classes_)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"    Train set: {X_train.shape}")
print(f"    Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("    ✓ Preprocessing complete")

# ============================================================
# 5. MODEL TRAINING & EVALUATION
# ============================================================
print("\n[5] Training and evaluating multiple models...")

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n    Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy
    }
    
    print(f"    ✓ Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"    ✓ Test accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n{'=' * 60}")
print(f"🏆 BEST MODEL: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")
print(f"{'=' * 60}")

# ============================================================
# 6. HYPERPARAMETER TUNING (Random Forest - best performer)
# ============================================================
print("\n[6] Hyperparameter tuning on Random Forest...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n    Best parameters: {grid_search.best_params_}")
print(f"    Best CV accuracy: {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"    Final test accuracy: {final_accuracy:.4f}")

# ============================================================
# 7. DETAILED EVALUATION
# ============================================================
print("\n[7] Detailed evaluation...")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=label_encoder.classes_))

# Confusion Matrix Heatmap
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved confusion_matrix.png")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved feature_importance.png")

# ============================================================
# 8. MODEL PERFORMANCE COMPARISON
# ============================================================
print("\n[8] Model performance comparison...")

comparison_df = pd.DataFrame([
    {'Model': name, 'CV Mean Accuracy': res['cv_mean'], 'Test Accuracy': res['test_accuracy']}
    for name, res in results.items()
])

plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df))
width = 0.35
plt.bar(x - width/2, comparison_df['CV Mean Accuracy'], width, label='CV Mean Accuracy', color='#2E86AB')
plt.bar(x + width/2, comparison_df['Test Accuracy'], width, label='Test Accuracy', color='#F18F01')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, comparison_df['Model'], rotation=45, ha='right')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('static/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved model_comparison.png")
print("\n" + comparison_df.to_string(index=False))

# ============================================================
# 9. K-MEANS CLUSTERING ANALYSIS
# ============================================================
print("\n[9] K-Means Clustering Analysis...")

# Determine optimal k using elbow method
inertias = []
K_range = range(1, 15)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', color='#2E86AB')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved elbow_method.png")

# Fit K-Means with k=22 (number of crop types)
kmeans = KMeans(n_clusters=len(label_encoder.classes_), random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
print(f"    ✓ K-Means clustering complete with k={len(label_encoder.classes_)}")

# ============================================================
# 10. SAVE MODELS & ENCODERS
# ============================================================
print("\n[10] Saving models and encoders...")

# Save the final tuned Random Forest model
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("    ✓ Saved crop_recommendation_model.pkl")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("    ✓ Saved scaler.pkl")

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("    ✓ Saved label_encoder.pkl")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print(f"   Best Model: Random Forest (Tuned)")
print(f"   Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"   Number of crops: {len(label_encoder.classes_)}")
print("=" * 60)
print(f"\nCrops supported: {', '.join(sorted(label_encoder.classes_))}")