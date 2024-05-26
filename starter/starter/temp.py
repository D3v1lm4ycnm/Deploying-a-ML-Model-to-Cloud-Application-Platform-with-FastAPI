import joblib
import pandas as pd
# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import compute_model_metrics, train_model



# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
# compute metrics
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)
print(X_test.shape)
print(y_test.shape)
 

