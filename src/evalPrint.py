import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evalP(y_test, y_pred, X, label_encoders, rf):
    
    print("Accuracy:", (accuracy_score(y_test, y_pred)*100), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    import matplotlib.pyplot as plt
    plt.show()

    print("\nEnter feature values to predict 'loan_status':")
    # Ask for all features
    input_cols = X.columns.tolist()

    sample = []
    for col in input_cols:
        prompt = f"Enter value for '{col}': "
        if col in label_encoders:
            categories = label_encoders[col].classes_
            prompt = f"Enter value for '{col}' {list(categories)}: "
        val = input(prompt + " ")
        try:
            val = float(val)
        except ValueError:
            pass
        sample.append(val)

    # Create a DataFrame for the input
    sample_df = pd.DataFrame([sample], columns=input_cols)

    # Encode categorical columns using label_encoders
    for col in label_encoders:
        if col in sample_df.columns:
            sample_df[col] = label_encoders[col].transform(sample_df[col])

    # Reorder columns to match training data
    sample_df = sample_df[X.columns]

    pred = rf.predict(sample_df)
    if 'loan_status' in label_encoders:
        pred_label = label_encoders['loan_status'].inverse_transform(pred)[0]
    else:
        pred_label = pred[0]
    print(f"\nPredicted 'loan_status': {pred_label}")