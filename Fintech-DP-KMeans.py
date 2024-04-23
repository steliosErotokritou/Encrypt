import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from diffprivlib.models import KMeans
import warnings

warnings.filterwarnings("ignore")

DIRECTORY = Path("./data")


def read_in_files():
    map = (
        pd.read_csv(DIRECTORY / "ca_relation_typed.csv")[["person_key", "account_key"]]
        .drop_duplicates("account_key", keep="first")
        .drop_duplicates("person_key", keep="first")
    )

    account = pd.read_csv(DIRECTORY / "account_typed.csv").join(
        map.set_index("account_key"), on="account_key"
    )
    account = account[~account.person_key.isna()]
    deposit = pd.read_csv(DIRECTORY / "deposit_account_typed.csv").join(
        map.set_index("person_key"), on="person_key"
    )
    deposit = deposit[~deposit.account_key.isna()]
    payment = pd.read_csv(DIRECTORY / "payment_typed.csv").join(
        map.set_index("account_key"), on="account_key"
    )
    payment = payment[~payment.person_key.isna()]
    person = pd.read_csv(DIRECTORY / "person_typed.csv").join(
        map.set_index("person_key"), on="person_key"
    )
    person = person[~person.account_key.isna()]

    return account, deposit, payment, person


def clean_files(account, deposit, payment, person):
    account = account.loc[:,
        [
            "account_key",
            "base_interest_rate",
            "repay_frequency",
            "number_of_total_installments",
            "delay_days",
            "overdue_expenses",
            "total_balance",
            "collateral_amount",
        ]
    ]

    min_delay = account.delay_days.min()
    max_delay = account.delay_days.max()

    account.delay_days = account.delay_days.apply(
        lambda x: ((x - min_delay) / (max_delay - min_delay)) * 210
    )

    deposit = deposit[
        [
            "account_key",
            "accounting_balance",
            "available_balance",
        ]
    ]

    payment = payment[["account_key", "capital_amount", "payinterest", "payexpenses"]]

    person = person[
        [
            "marital_status",
            "gender",
            "account_key",
        ]
    ]

    return account, deposit, payment, person


def join_files(account, deposit, payment, person):
    df = (
        (
            account.join(deposit.set_index("account_key"), on="account_key")
            .join(payment.set_index("account_key"), on="account_key")
            .join(person.set_index("account_key"), on="account_key")
        )
        .drop("account_key", axis=1)
        .reset_index(drop=True)
    )

    df.marital_status.fillna("single", inplace=True)
    df.gender.fillna("M", inplace=True)
    df.fillna(0, inplace=True)

    return df


def feature_engineering(df):
    categorical_columns = ["gender", "marital_status"]
    categories = pd.get_dummies(df[categorical_columns], drop_first=True)
    df[categories.columns] = categories
    df.drop(categorical_columns, axis=1, inplace=True)

    columns_to_scale = [
        "base_interest_rate",
        "repay_frequency",
        "number_of_total_installments",
        "overdue_expenses",
        "total_balance",
        "collateral_amount",
        "accounting_balance",
        "available_balance",
        "capital_amount",
        "payinterest",
        "payexpenses",
    ]

    scaler = StandardScaler().fit(df[columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    return df


def fit_model(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, bounds=(0, 1)).fit(df)
    return kmeans


def return_accuracy(kmeans, df):
    labels = kmeans.predict(df)

    # Example of assigning labels based on majority class
    # You may need to adjust this based on your specific dataset
    assigned_labels = [1 if df[df.index == i]['delay_days'].mean() > 180 else 0 for i in labels]

    accuracy = accuracy_score(df['target_variable'], assigned_labels)

    return accuracy


if __name__ == "__main__":
    accounts, deposits, payments, persons = read_in_files()
    accounts, deposits, payments, persons = clean_files(
        accounts, deposits, payments, persons
    )
    full_df = join_files(accounts, deposits, payments, persons)
    full_df = feature_engineering(full_df)
    
    # Assuming 'delay_days' is the target variable
    full_df["target_variable"] = full_df.delay_days.apply(lambda x: 1 if x > 180 else 0)
    full_df.drop("delay_days", axis=1, inplace=True)
    
    # Specify the number of clusters for K-Means
    n_clusters = 2

    model = fit_model(full_df, n_clusters)
    acc = return_accuracy(model, full_df)
    print(f"The accuracy of the model was: {acc:.4f}")
