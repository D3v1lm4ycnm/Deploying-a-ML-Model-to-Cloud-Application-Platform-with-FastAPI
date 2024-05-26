import joblib
import pandas as pd
# Script to train machine learning model.
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import compute_model_metrics, train_model


# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)
# kfold = KFold(n_splits=5)
# for train, test in kfold.split(data):
#     train = data.iloc[train]
#     test = data.iloc[test]

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train model.
print("Training model")
model = train_model(X_train, y_train)

# compute metrics
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
y_pred = model.predict(X_test)
precision, recal, fbeta = compute_model_metrics(y_pred, y_test)

print(f"Precision: {precision}")
print(f"Recall: {recal}")
print(f"F1: {fbeta}")

# Save the model and encoder.
print("Saving model")
joblib.dump(model, "starter/model/model.pkl")
joblib.dump(encoder, "starter/model/encoder.pkl")
joblib.dump(lb, "starter/model/lb.pkl")
