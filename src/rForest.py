from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def rfc_model(df, label_encoders):

    # Prepare features and target
    X = df.drop('visa eligible', axis=1)
    y = df['visa eligible']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the model
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    return y_pred, y_test, X_test, label_encoders, rf