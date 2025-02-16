# Customer Churn Prediction using CNN & Ensemble Learning

## 📌 Project Overview
This project aims to predict **customer churn** using a dataset containing customer demographics, usage patterns, and service details. The model leverages **deep learning (CNN)** and **ensemble machine learning** techniques for robust classification.

## 📂 Dataset
- **File Name:** `customer_churn_dataset-testing-master.csv`
- **Target Variable:** `Churn` (1 = Churned, 0 = Retained)
- **Features:**
  - Categorical: `Gender`, `Subscription Type`, `Contract Length`
  - Numerical: `Age`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`

## 🔧 Data Preprocessing
- **Handling Categorical Data:** Used `LabelEncoder` to convert categorical variables into numerical form.
- **Feature Scaling:** Applied `StandardScaler` to normalize numerical values.
- **Train-Test Split:** Split dataset into **80% training** and **20% testing**.
- **Reshaping for CNN:** Converted feature matrix into 3D format for **Conv1D** processing.

## 📊 Exploratory Data Analysis (EDA)
- **Distribution of Churn Classes** (Visualization using Seaborn)
- **Correlation Heatmap** (Visualizing feature relationships)
- **Boxplots & Histograms** (For feature distribution analysis)

## 🧠 Machine Learning Models Used
1. **Convolutional Neural Network (CNN)**
   - Conv1D layers with Batch Normalization & Dropout
   - Fully connected Dense layers for classification
   - **Activation Function:** ReLU & Sigmoid
   - **Optimizer:** Adam

2. **Ensemble Learning Models:**
   - **Random Forest** (n_estimators=100, parallelized with `n_jobs=-1`)
   - **Gradient Boosting** (n_estimators=50, learning_rate=0.1)
   - **AdaBoost** (n_estimators=100)
   - **Support Vector Machine (SVM)** (with linear and RBF kernel variations)
   - **Voting Classifier** (Combining RF, GB, AdaBoost, SVM for better prediction)

## 📈 Model Evaluation
- **CNN Performance:** Accuracy on test set displayed after training.
- **Classification Reports:** Precision, Recall, and F1-score for all models.
- **Confusion Matrices:** Heatmaps generated for performance visualization.

## 🚀 How to Run the Project
### Prerequisites
Install required Python libraries:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

### Run the Jupyter Notebook
Execute the notebook (`customer_churn.ipynb`) step by step:
```bash
jupyter notebook customer_churn.ipynb
```

### Expected Output
- Model training logs
- Performance metrics (accuracy, precision, recall, F1-score)
- EDA visualizations and confusion matrices

## 📌 Future Enhancements
- **Hyperparameter Tuning** for CNN and ML models.
- **Feature Engineering** to improve model performance.
- **Deploying Model** via Flask or FastAPI for real-world use.

---
