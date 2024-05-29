import pandas as pd

df = pd.read_csv("data/census.csv")


def slice(df, features, filename):
    """ Function for calculating descriptive stats on slices based on feature.
    Input:
        df - pandas DataFrame
        features - list of features to slice on
        filename - name of the file to write the results to
    """
    with open(filename, "w") as f:
        for feature in features:
            for cls in df["salary"].unique():
                df_temp = df[df["salary"] == cls]
                mean = df_temp[feature].value_counts().mean()
                stddev = df_temp[feature].value_counts().std()
                # mean = df_temp[feature].mean()
                # stddev = df_temp[feature].std()
                f.writelines(f"Class: {cls}")
                f.writelines("\n")
                f.writelines(f"{feature} mean: {mean:.4f}")
                f.write(", ")
                f.writelines(f"{feature} stddev: {stddev:.4f}")
                f.writelines("\n")


if __name__ == "__main__":
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
    slice(df, cat_features, "slice_output.txt")
