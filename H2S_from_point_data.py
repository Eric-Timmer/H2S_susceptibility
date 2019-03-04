import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing




def pre_process(df):
    """
    replace #N/A with np.NaN
    replace <Null> with np.NaN

    remove extra columns

    separate class from predictors

    convert categorical to numerical

    :param df:
    :return processed df:
    """
    # replace certain values
    df.replace(to_replace="#N/A", value=np.NaN, inplace=True)
    df.replace(to_replace="<Null>", value=np.NaN, inplace=True)
    df.replace(to_replace="NULL", value=np.NaN, inplace=True)

    y = df["H2S_binary"]
    df.drop(columns=["H2S_binary"], inplace=True)
    print(df.columns)
    return df, y


def standardize(X):
    predictors = X.columns

    # # remove mean and standard deviation
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = preprocessing.minmax_scale(X)

    # # convert back to dataframe for clearer results
    X = pd.DataFrame(X, columns=predictors)
    return X

def logistic_regression(X, y):
    print(X.index, 'b', y.index)
    logit_model = sm.Logit(y, X, missing="drop")
    print(logit_model)
    results = logit_model.fit_regularized()
    print(results.summary())


def main():
    # load data
    path = r""
    df = pd.read_excel(path)
    X, y = pre_process(df)

    scaled_X = standardize(X)
    print(scaled_X.index, y.index)

    logistic_regression(scaled_X, y)

    return

if __name__ == "__main__":
    main()