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

    print("\nEnter feature values to predict 'visa eligible':")
    # Only ask for features up to 'grade'
    input_cols = []
    for col in X.columns:
        input_cols.append(col)
        if col == 'ielts_group':
            break

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

    # Recompute engineered features for the sample if needed
    if {'math score', 'reading score', 'writing score'}.issubset(sample_df.columns):
        sample_df['percentage'] = ((sample_df['math score'] + sample_df['reading score'] + sample_df['writing score']) / 300) * 100
    if 'percentage' in sample_df.columns:
        sample_df['grade'] = np.where(sample_df['percentage'] >= 85, 'A',
                            np.where(sample_df['percentage'] >= 70, 'B',
                            np.where(sample_df['percentage'] >= 55, 'C', 'F')))
    if 'age' in sample_df.columns:
        sample_df['age_group'] = np.where(sample_df['age'] < 18, 'Below 18',
                        np.where(sample_df['age'] < 20, '18-19',
                        np.where(sample_df['age'] < 22, '20-21',
                        np.where(sample_df['age'] < 24, '22-23', '24 and above'))))
    if 'parental level of education' in sample_df.columns:
        edu_order = {
            "some high school": 0,
            "high school": 1,
            "some college": 2,
            "associate's degree": 3,
            "bachelor's degree": 4,
            "master's degree": 5
        }
        sample_df['parental_education_ord'] = sample_df['parental level of education'].map(edu_order)
    if 'financial sponsorship' in sample_df.columns:
        sample_df['is_scholarship'] = (sample_df['financial sponsorship'].str.lower() == 'scholarship').astype(int)

    # Encode categorical columns using label_encoders
    for col in label_encoders:
        if col in sample_df.columns:
            sample_df[col] = label_encoders[col].transform(sample_df[col])

    # Reorder columns to match training data
    sample_df = sample_df[X.columns]

    pred = rf.predict(sample_df)
    if 'visa eligible' in label_encoders:
        pred_label = label_encoders['visa eligible'].inverse_transform(pred)[0]
    else:
        pred_label = pred[0]
    print(f"\nPredicted 'visa eligible': {pred_label}")