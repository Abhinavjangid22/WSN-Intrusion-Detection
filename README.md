# WSN Intrusion Detection System

## This Project
- Uses **NSL-KDD dataset** for training and testing.
- Implements **Random Forest**, **SVM**, and **XGBoost** classifiers.
- Encodes categorical features using **One-Hot Encoding**.
- Applies **feature scaling** for SVM for better performance.
- Handles imbalanced data using **class weights**.
- Reports **Accuracy, Precision, Recall, and F1-score** for evaluation.
- Visualizes results using **Confusion Matrix**.

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` for data manipulation  
  - `numpy` for numerical computations  
  - `scikit-learn` for ML models and metrics  
  - `xgboost` for Gradient Boosting classifier  

## Project Structure
WSN-Intrusion-Detection/
│
├── data/
│ ├── KDDTrain+.TXT # Training dataset
│ └── KDDTest+.TXT # Testing dataset
│
├── wsn_ids.py # Main ML script
├── README.md # Project description
└── requirements.txt # Python dependencies


## How It Works
1. **Load Dataset:** Training and testing sets from `data/` folder.
2. **Feature Encoding:** Categorical features converted to numeric using **One-Hot Encoding**.
3. **Scaling:** `StandardScaler` applied for SVM model.
4. **Model Training:**
   - Random Forest (`class_weight='balanced'`)
   - SVM (RBF kernel, `class_weight='balanced'`)
   - XGBoost (`eval_metric='mlogloss'`)
5. **Evaluation:** Prints **Accuracy**, **Classification Report**, and plots **Confusion Matrix** for Random Forest.

## ⚙ How It Works
1. **Load Dataset:** Training and testing sets from `data/` folder.
2. **Feature Encoding:** Categorical features converted to numeric using One-Hot Encoding.
3. **Scaling:** StandardScaler applied for SVM model.
4. **Model Training:**
   - Random Forest (`class_weight='balanced'`)
   - SVM (`rbf` kernel, `class_weight='balanced'`)
   - XGBoost (`eval_metric='mlogloss'`)
5. **Evaluation:** Prints Accuracy, Classification Report, and plots Confusion Matrix for Random Forest.

## 🚀 How to Run
```bash
# 1. Clone the repository

git clone https://github.com/Abhinavjangid22/WSN-Intrusion-Detection.git
cd WSN-Intrusion-Detection

# 2. Install dependencies

pip install -r requirements.txt

# 3. Place NSL-KDD dataset files in data/ folder
#    - KDDTrain+.TXT
#    - KDDTest+.TXT

# 4. Run the script
python wsn_ids.py

# 5. Check console output for:
#    - Accuracy
#    - F1-score
#    - Confusion Matrix plot

# 6. 📊 Sample Output
# === Random Forest Results ===
# Accuracy: 0.63
# Weighted F1-score: 0.58
# Classification report for all attack classes printed.
# Confusion matrix plotted.

# 7. ⚡ Future Improvements
# - Apply SMOTE or other oversampling methods to balance minority attack classes.
# - Use Deep Learning models (e.g., LSTM, CNN) for better detection.
# - Hyperparameter tuning via GridSearchCV / RandomizedSearchCV.
# - Real-time WSN intrusion detection using streaming network data.

# 8. 📜 References
# - NSL-KDD Dataset
# - Scikit-learn documentation: https://scikit-learn.org
# - XGBoost documentation: https://xgboost.readthedocs.io
uirements.txt**
