import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

acc_dt_unconstrained_train = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
acc_dt_unconstrained_test = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))

print("=== Task 1: Unconstrained Decision Tree ===")
print(f"Train Accuracy: {acc_dt_unconstrained_train:.4f}")
print(f"Test Accuracy:  {acc_dt_unconstrained_test:.4f}\n")

dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

acc_dt_controlled_train = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
acc_dt_controlled_test = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))

print("=== Task 2: Controlled Decision Tree ===")
print(f"Train Accuracy: {acc_dt_controlled_train:.4f}")
print(f"Test Accuracy:  {acc_dt_controlled_test:.4f}\n")

dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)

dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)

print("=== Task 3: Criterion Comparison ===")
print(f"Gini Test Accuracy:    {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)):.4f}")
print(f"Entropy Test Accuracy: {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)):.4f}\n")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_test_scaled)[:, 1])

print("=== Task 4: Random Forest ===")
print(f"Train Accuracy: {rf_train_acc:.4f} | Test Accuracy: {rf_test_acc:.4f} | Test AUC: {rf_test_auc:.4f}")

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 5 Features by Importance:")
for i in range(5):
    print(f"Feature Index {indices[i]}: {importances[indices[i]]:.4f}")
print()

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_test_scaled)[:, 1])

print("=== Task 4a: Gradient Boosting ===")
print(f"Train Accuracy: {gb_train_acc:.4f} | Test Accuracy: {gb_test_acc:.4f} | Test AUC: {gb_test_auc:.4f}\n")

lowest_5_indices = indices[-5:]

if isinstance(X_train, pd.DataFrame):
    lowest_5_features = [X_train.columns[idx] for idx in lowest_5_indices]
    X_train_reduced = pd.DataFrame(X_train_scaled, columns=X_train.columns).drop(columns=lowest_5_features)
    X_test_reduced = pd.DataFrame(X_test_scaled, columns=X_test.columns).drop(columns=lowest_5_features)
else:
    keep_indices = [i for i in range(X_train_scaled.shape[1]) if i not in lowest_5_indices]
    X_train_reduced = X_train_scaled[:, keep_indices]
    X_test_reduced = X_test_scaled[:, keep_indices]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print("=== Task 4b: Feature Ablation ===")
print(f"Full Model Test AUC:    {rf_test_auc:.4f}")
print(f"Reduced Model Test AUC: {rf_reduced_auc:.4f}\n")

lr_model = LogisticRegression(random_state=42)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": lr_model,
    "Controlled Decision Tree": dt_controlled,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

print("=== Task 5: Cross-Validated Comparison (AUC) ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc')
    print(f"{name} -> Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")
print()

pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_clf_train)

print("=== Task 6: GridSearchCV Results ===")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV AUC Score: {grid_search.best_score_:.4f}\n")

best_pipeline = grid_search.best_estimator_
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

print("=== Task 7: Manual Learning Curve Table ===")
print(f"{'Training fraction':<18} | {'Training AUC':<12} | {'Test AUC':<8}")
print("-" * 46)

for f in fractions:
    n_samples = int(f * len(X_train))
    
    if isinstance(X_train, pd.DataFrame):
        X_sub = X_train.iloc[:n_samples]
    else:
        X_sub = X_train[:n_samples]
        
    y_sub = y_clf_train[:n_samples]
    
    best_pipeline.fit(X_sub, y_sub)
    
    train_auc = roc_auc_score(y_sub, best_pipeline.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    
    print(f"{f:<18.1f} | {train_auc:<12.4f} | {test_auc:.4f}")
print()

joblib.dump(best_pipeline, 'best_model.pkl')

print("=== Task 8: Verification of Serialization ===")
loaded_model = joblib.load('best_model.pkl')

if isinstance(X_train, pd.DataFrame):
    hand_crafted_data = pd.DataFrame(np.random.randn(2, X_train.shape[1]), columns=X_train.columns)
else:
    hand_crafted_data = np.random.randn(2, X_train.shape[1])

predictions = loaded_model.predict(hand_crafted_data)
print(f"Predictions for hand-crafted rows: {predictions}")
