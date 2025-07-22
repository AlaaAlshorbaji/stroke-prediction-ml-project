# 🧠 Stroke Prediction using Machine Learning

This project focuses on predicting whether a patient is likely to suffer a **stroke** using demographic and health-related data. Early stroke prediction can save lives and reduce healthcare costs.

We build a machine learning pipeline that includes:
- Data exploration (EDA)
- Preprocessing and cleaning
- Handling class imbalance
- Training and evaluating classifiers

---

## 🎯 Objective

To classify individuals at risk of having a **stroke** based on features such as:
- Age
- Hypertension
- Heart disease
- BMI
- Work type
- Residence type
- Smoking status

---

## 📊 Dataset Description

The dataset contains the following features:

| Feature              | Description                                  |
|----------------------|----------------------------------------------|
| id                   | Unique identifier                            |
| gender               | Male, Female, or Other                       |
| age                  | Age of the patient                           |
| hypertension         | 1 if the patient has hypertension            |
| heart_disease        | 1 if the patient has any heart disease       |
| ever_married         | Marital status                               |
| work_type            | Type of employment                           |
| Residence_type       | Urban or Rural                               |
| avg_glucose_level    | Average glucose level in blood               |
| bmi                  | Body Mass Index                              |
| smoking_status       | Never smoked, formerly smoked, etc.          |
| stroke               | Target variable (1 = stroke, 0 = no stroke)  |

---

## 📦 Libraries Used

- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib`, `seaborn` – Data visualization
- `scikit-learn` – Modeling, evaluation, and preprocessing
- `imblearn` – SMOTE for class imbalance
- `google.colab` – Compatibility for Google Colab

---

## 🔍 Step-by-Step Workflow

### 1. 📥 Importing Libraries & Loading Data
- Loaded dataset using `pandas.read_csv()`
- Checked basic structure and missing values

### 2. 📈 Exploratory Data Analysis (EDA)
- **Target variable distribution** to check class imbalance
- **Count plots** for categorical features (e.g., gender, marital status)
- **Histograms** for numeric variables (e.g., age, glucose)
- **Boxplots** grouped by stroke status
- **Heatmap** for correlation analysis

### 3. 🧹 Data Preprocessing
- Converted `bmi` to numeric and handled missing values
- Categorical encoding using LabelEncoder and One-Hot
- Normalization/Standardization of numeric values if needed

### 4. ⚖️ Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) from `imblearn` to generate synthetic samples of minority class (stroke = 1)

### 5. 🤖 Model Training
Trained multiple classifiers:
- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

Used `train_test_split()` to split data into training and testing sets.

### 6. 📐 Evaluation Metrics
For each model:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC-AUC Curve**

Also used classification reports and visual comparisons.

---

## 🚀 How to Run

1. Download or clone this repository.
2. Open `Stroke_Prediction_Dataset.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/).
3. Run all cells sequentially.
4. Required packages:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
