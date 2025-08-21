import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load dataset
# -------------------------
train_df = pd.read_csv("data/KDDTrain+.TXT", header=None)
test_df = pd.read_csv("data/KDDTest+.TXT", header=None)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# -------------------------
# 2. Encode categorical features
# -------------------------
categorical_cols = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Align test set columns with train set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Convert column names to string (scikit-learn fix)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# -------------------------
# 3. Scaling for SVM
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 4. Random Forest Classifier
# -------------------------
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

print("=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))

# -------------------------
# 5. SVM Classifier
# -------------------------
svm_clf = SVC(kernel='rbf', class_weight='balanced')
svm_clf.fit(X_train_scaled, y_train)
svm_pred = svm_clf.predict(X_test_scaled)

print("=== SVM Results ===")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, zero_division=0))

# -------------------------
# 6. XGBoost Classifier
# -------------------------
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)

print("=== XGBoost Results ===")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, zero_division=0))

# -------------------------
# 7. Confusion Matrix Plot (Random Forest)
# -------------------------
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
