import numpy as np
import pandas as pd
from sklearn import datasets

def load_dataset():
    # Load the digits dataset (0–9)
    digits = datasets.load_digits()
    X = pd.DataFrame(digits.data)
    print(X)
    y = pd.Series(digits.target)

    # Add target to the dataframe
    X['label'] = y

    X = X.dropna(axis=1, how='all')                 # drop all-NaN columns
    X = X.loc[:, ~(X == 0).all()]                  # drop all-0 columns


    return X  # Return DataFrame with features + label


if __name__ == "__main__":
    df = load_dataset()
    print(df)
    df.to_csv("digits.csv", index=False)  # Save to CSV without row indices
    print("CSV file saved as digits.csv")
