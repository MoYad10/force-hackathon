import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(df, file_io):
    df = df.dropna()
    df = df.dropna(axis=1, how="all")
    df = df.drop(["Unnamed: 0", "timestamp", "SKAP_18HV3806/BCH/10sSAMP|stepinterpolation"], axis=1)

    output_tags = [
        "SKAP_18FI381-VFlLGas/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLH2O/Y/10sSAMP|average",
        "SKAP_18FI381-VFlLOil/Y/10sSAMP|average",
    ]

    y_tags = ["SKAP_18FI381-VFlLGas/Y/10sSAMP|average"]

    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)

    X_train = df_train.drop(output_tags, axis=1)
    y_train = df_train[y_tags]
    y_train.columns = ["gas"]

    X_test = df_test.drop(output_tags, axis=1)
    y_test = df_test[y_tags]
    y_test.columns = ["gas"]

    # Scale
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    with file_io("y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    with file_io("X_scaler.pkl", "wb") as f:
        pickle.dump(X_scaler, f)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
