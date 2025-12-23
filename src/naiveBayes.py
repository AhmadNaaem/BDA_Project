from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def nb_model(X_train, X_test, y_train, y_test, label_encoders):
    dt = GaussianNB()
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    y_score = dt.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    return y_pred, y_test, y_score, X_test, label_encoders, dt, acc
