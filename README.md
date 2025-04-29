# Predicting-Hospital-Readmissions

### 📚 Project Overview

Hospital readmissions are a major concern in healthcare, affecting both patient outcomes and hospital costs. This project focuses on predicting patient readmissions using clinical and demographic data, with a special emphasis on diabetes-related indicators.
We apply a range of machine learning techniques to build predictive models and interpret the factors driving readmissions.

### 🛠️ Technologies Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Google Colab
- Machine Learning Models: Logistic Regression, Random Forest, XGBoost, Histogram-Based Gradient Boosting, Multi-Layer Perceptron (MLP)
- Hyperparameter Tuning: GridSearchCV
- Evaluation Metrics: F1-Score, Precision, Recall, ROC-AUC

### 📈 Approach

##### Data Cleaning and Preprocessing:

- One-hot encoding of categorical variables
- Label encoding for binary target variable (readmitted)
- Min-Max Scaling and Standardization of numerical features
- Train/Validation/Test split with a predefined split strategy
  
##### Exploratory Data Analysis (EDA):

- Distribution analysis of target variable
- Correlation matrix of numerical features
- Boxplots to detect outliers and feature behavior by readmission status
- Feature importance extraction using Random Forest

##### Model Building and Hyperparameter Tuning:

- Applied and fine-tuned multiple models using GridSearchCV
- Evaluation based on macro-averaged F1-score to handle moderate class imbalance
  
##### Final Model Evaluation

- Selected Random Forest Classifier as the best model
- Achieved Precision: 65.6%, Recall: 62.7%, F1-Score: 64.1%, AUC: 0.742 on validation set
- Analyzed Confusion Matrix and ROC Curve
  
### 🧠 Key Insights

- Lab procedures, number of medications, and time in hospital were the strongest predictors of readmission.
- Tree-based ensemble methods like Random Forest and XGBoost performed better than simple models or neural networks for this structured healthcare dataset.
- Moderate class imbalance was handled using class weighting without the need for oversampling.

### 🚀 Future Work

- Explore ensemble stacking or boosting of multiple models
- Investigate time-series patient history features
- Apply cost-sensitive learning to further minimize false positives in clinical predictions
