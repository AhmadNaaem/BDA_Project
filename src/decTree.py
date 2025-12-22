from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def tree_model(df, label_encoders):

    # Prepare features and target
    X = df.drop('visa eligible', axis=1)
    y = df['visa eligible']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42)

    # Train the model
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print("Accuracy:", (accuracy_score(y_test, y_pred)*100), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    
    print("\nEnter feature values to predict 'visa eligible':")
    sample = []
    for col in X.columns:
        prompt = f"Enter value for '{col}': "
        # If column is label encoded, show possible categories
        if col in label_encoders:
            categories = label_encoders[col].classes_
            prompt = f"Enter value for '{col}' {list(categories)}: "
        val = input(prompt + " ")

        try:
            val = float(val)
        except ValueError:
            pass
        sample.append(val)
    sample_df = pd.DataFrame([sample], columns=X.columns)

    for col in label_encoders:
        if col in sample_df.columns:
            sample_df[col] = label_encoders[col].transform(sample_df[col])
    pred = dt.predict(sample_df)
    if 'visa eligible' in label_encoders:
        pred_label = label_encoders['visa eligible'].inverse_transform(pred)[0]
    else:
        pred_label = pred[0]
    print(f"\nPredicted 'visa eligible': {pred_label}")