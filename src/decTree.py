from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def tree_model(df, label_encoders):

    # Prepare features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42)
    # dt = DecisionTreeClassifier(max_depth=7,min_samples_split=20,min_samples_leaf=10,class_weight='balanced',random_state=42)

    # Train the model
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)


    y_score = dt.predict_proba(X_test)[:, 1] 
    acc = accuracy_score(y_test, y_pred)

    
    return y_pred, y_test,y_score, X_test, label_encoders, dt,acc