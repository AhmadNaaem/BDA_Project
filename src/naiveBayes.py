from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def nb_model(df, label_encoders):

    # Prepare features and target (replace 'target' with your actual target column)
    X = df.drop('visa eligible', axis=1)
    y = df['visa eligible']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    nb = GaussianNB()

    # Train the model
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print("Accuracy:", (accuracy_score(y_test, y_pred)*100), "%")    
    
    print("\nEnter feature values to predict 'visa eligible':" )
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
    pred = nb.predict(sample_df)
    if 'visa eligible' in label_encoders:
        pred_label = label_encoders['visa eligible'].inverse_transform(pred)[0]
    else:
        pred_label = pred[0]
    print(f"\nPredicted 'visa eligible': {pred_label}")

