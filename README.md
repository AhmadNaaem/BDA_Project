# BDA_Project
Project Name: Credit Risk Prediction System

## Description
This project uses machine learning to predict whether a loan applicant is likely to default based on their financial and demographic profile data. It includes data analysis, feature engineering, and multiple machine learning models (Naive Bayes, Decision Tree, Random Forest). The project also provides a GUI for interactive EDA, model evaluation, ROC curve comparison, and making predictions.

## Structure
- src/main.py - Main script to run the project
- src/EDA.py - Exploratory Data Analysis functions and visualizations
- src/featureEngineering.py - Feature engineering logic
- src/naiveBayes.py - Naive Bayes model training
- src/decTree.py - Decision Tree model training
- src/rForest.py - Random Forest model training
- src/evalPrint.py - Model evaluation and prediction utilities
- src/ROC_curve.py - ROC curve plotting and comparison
- src/GUI.py - Graphical User Interface for EDA, evaluation, and prediction
- src/credit_risk.csv - Dataset

## Dataset Features
The model uses the following features for prediction:
- person_income - Annual income of the borrower
- person_home_ownership - Home ownership status (encoded)
- loan_intent - Purpose of the loan (encoded)
- loan_grade - Credit grade of the loan (encoded)
- loan_amnt - Loan amount requested
- cred_history_year - Years of credit history (engineered feature)
- age_group - Age category of the borrower (engineered feature)

Target Variable: loan_status (0 = Non-default, 1 = Default)

## How to Run
Install requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the main script:
```bash
python src/main.py
```

## Features
- Interactive GUI for EDA, model evaluation, and predictions
- Multiple ML models: Naive Bayes, Decision Tree, Random Forest
- Visualizations: categorical distributions, heatmaps, confusion matrix, ROC curves
- Feature engineering and data preprocessing
- Model comparison using accuracy metrics and ROC-AUC curves
- Real-time loan default prediction based on borrower profile

## Technologies Used
- Python 3.11.9
- scikit-learn - Machine learning algorithms
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib & seaborn - Data visualization
- tkinter - GUI development

## Author
Ahmad Naaem Saad
