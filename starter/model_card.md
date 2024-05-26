# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest Classifier using GridSearchCV for parameters tunning (Scikit-Learn)

## Intended Use
Predict income base on "workclass", "education", "marital-status", "occupation", "relationship" , "race", "sex", "native-country"

## Training Data
80% of census dataset

## Evaluation Data
20% of census dataset

## Metrics
precesion, recall and F1 score are use for evaluate model
- Precision: 0.531
- Recall: 0.776
- F1: 0.631

## Caveats and Recommendations
The dataset contains many of feature based sex, country, occupation, race are not fairly.
