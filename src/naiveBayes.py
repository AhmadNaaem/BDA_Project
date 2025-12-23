from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def nb_model(df, label_encoders):

    # Prepare features and target (replace 'target' with your actual target column)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    nb = GaussianNB()

    # Train the model
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    y_score = nb.predict_proba(X_test)[:, 1] 
    acc = accuracy_score(y_test, y_pred)

    
    return y_pred, y_test,y_score, X_test, label_encoders, nb,acc