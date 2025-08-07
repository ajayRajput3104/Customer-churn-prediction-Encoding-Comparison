# 🔍 Telco Customer Churn Prediction

This project analyzes the Telco Customer Churn dataset to predict whether a customer will churn based on their demographic and service usage data. We explore **different categorical encoding techniques** and compare their **impact on model performance** using XGBoost.

---

## 📁 Project Structure
```
.
├── Telco_Customer_Churn.ipynb # Main notebook with code, EDA, model training & evaluation
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset
├── outputs/
│ ├── performance_plots/ # Bar plots comparing precision, recall, f1, accuracy
│ └── confusion_matrices/ # Confusion matrices for each model
└── README.md # This file
```

```yaml


---

## 📊 Dataset Overview

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7043 customers  
- **Columns**: 21 features including:
  - **Demographic**: Gender, SeniorCitizen, Partner, Dependents
  - **Service**: InternetService, StreamingTV, OnlineSecurity, etc.
  - **Account**: Tenure, MonthlyCharges, TotalCharges, PaymentMethod
  - **Target**: `Churn` (0 = No, 1 = Yes)

---

## ⚙️ Objective

To evaluate how different encoding methods affect the predictive performance of the model:
- **Binary Encoding**
- **Label Encoding**
- **One-Hot Encoding**
- **Mixed (Best-suited combination)**

---

## 🧪 Encoding Techniques Used

| Column Type        | Columns                                                                                  | Encoding Technique |
|--------------------|-------------------------------------------------------------------------------------------|---------------------|
| Binary             | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`                              | Binary (Yes/No → 1/0) |
| Multi-class (Nominal) | `gender`, `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`            | One-Hot or Label     |
| Service columns with "No internet service" | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Binary or Label      |

---

## 🤖 Model Details

- **Model Used**: XGBoost Classifier (`binary:logistic`)
- **Training Strategy**:
  - Train/test split (80/20), stratified
  - Evaluated on: Accuracy, Precision, Recall, F1 Score
  - GPU acceleration used (`tree_method='hist'`, `device='cuda'`)

---

## 📈 Performance Comparison

| Encoding Type | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| **Binary**     | 0.7896   | 0.6154    | 0.5561 | 0.5843   |
| **One-Hot**    | 0.7811   | 0.6006    | 0.5267 | 0.5613   |
| **Label**      | 0.7818   | 0.6012    | 0.5321 | 0.5645   |
| **Mixed**      | 0.7839   | 0.6061    | 0.5348 | 0.5682   |

---

## 🧾 Observations

- Binary encoding slightly outperformed others overall.
- Mixed encoding performed consistently across all metrics, offering a strong balance.
- One-hot encoding created the largest number of features, which may impact scalability on larger datasets.
- Label encoding can mislead tree-based models if there's **ordinal implication** in categories (which isn't true here).

---

## 📸 Visual Outputs

> All plots are saved in the `outputs/` folder.

### 📊 Performance Metrics

Located in: `outputs/performance_plots/`
- F1 Score comparison bar graph
- Precision comparison bar graph
- Recall comparison bar graph
- Accuracy comparison bar graph

### 🔲 Confusion Matrices

Located in: `outputs/confusion_matrices/`
- Binary Encoding
- Label Encoding
- One-Hot Encoding
- Mixed Encoding

---
```
## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/telco-churn-encoding-analysis.git
   cd telco-churn-encoding-analysis
   ```
Install required libraries:

```bash
pip install -r requirements.txt
```
Open the notebook:
``` bash
jupyter notebook Telco_Customer_Churn.ipynb
```
📚 Future Improvements
-Hyperparameter tuning using Optuna or GridSearchCV

-Try other encodings: Target Encoding, Frequency Encoding

-Test different models: RandomForest, LightGBM, CatBoost

-Feature selection based on SHAP values

📬 Contact
For queries or collaborations, reach out via GitHub Issues or ping me on LinkedIn.


