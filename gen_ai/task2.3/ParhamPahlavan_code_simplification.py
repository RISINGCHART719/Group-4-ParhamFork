import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Read dataset
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Encode categorical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
data[categorical_features] = data[categorical_features].apply(LabelEncoder().fit_transform)

# Define dependent and independent variables
y = data['NObeyesdad']
X = data.drop('NObeyesdad', axis=1).to_numpy()

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=90, random_state=20, max_depth=25)

# PCA for visualization
pca = PCA(n_components=2)

# Define class colors dynamically
unique_labels = y.unique()
colors = ['r', 'b', 'g', 'orange', 'purple', 'black', 'pink']
weight_colors = dict(zip(unique_labels, colors))

# Create legend handles
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in weight_colors.items()]

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=95)
accuracy_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train and evaluate model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    accuracy_scores.append(accuracy)

    # PCA visualization
    X_train_pca = pca.fit_transform(X_train)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=[weight_colors[label] for label in y_train], alpha=0.7)
    plt.title(f'PCA Training Data Visualization for Fold {fold}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles=legend_handles, title="Weights")
    plt.show()

# Print accuracy results
fold_labels = [f'Fold {i}' for i in range(1, 11)]
accuracy_df = pd.DataFrame([accuracy_scores], columns=fold_labels)
print(accuracy_df)

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.bar(fold_labels, accuracy_scores, color='maroon', width=0.65)
plt.ylim(0.5, 1)
plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy of Each Fold')
plt.bar_label(plt.bar(fold_labels, accuracy_scores), labels=[f'{score:.4f}' for score in accuracy_scores], fontsize=8)
plt.show()

# Calculate and display average accuracy
print('Average Fold Accuracy:', np.mean(accuracy_scores))
