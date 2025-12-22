from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def ts_data_split(
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        eval_start:str,
        eval_end:str
        ):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)]
    eval =  df.loc[(df["date"] >= eval_start) & (df["date"] <= eval_end)]

    return train , eval


def fit_customer_segment_ohe(df, cat_cols):
    encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
    encoder.fit(df[cat_cols])
    return encoder

def customer_segment_ohe(df, encoder, cat_cols):
    ohe = encoder.transform(df[cat_cols])
    cols = encoder.get_feature_names_out(cat_cols).tolist()

    ohe_df = pd.DataFrame(ohe, columns=cols, index=df.index)
    df = pd.concat([df, ohe_df], axis=1)

    rename_map = {
        'dept_id_FOODS_1': 'FOODS_1',
        'dept_id_FOODS_2': 'FOODS_2',
        'dept_id_FOODS_3': 'FOODS_3',
        'dept_id_HOBBIES_1': 'HOBBIES_1',
        'dept_id_HOBBIES_2': 'HOBBIES_2',
        'dept_id_HOUSEHOLD_1': 'HOUSEHOLD_1',
        'dept_id_HOUSEHOLD_2': 'HOUSEHOLD_2',
        'customer_segment_0': 'segment_0',
        'customer_segment_1': 'segment_1',
        'customer_segment_2': 'segment_2',
        'customer_segment_3': 'segment_3',
    }
    df.rename(columns=rename_map, inplace=True)
    cols = [rename_map.get(c, c) for c in cols]

    return df, cols


import pandas as pd

def add_lag_feature(
    df: pd.DataFrame,
    value_col: str,
    group_col: list= ['dept_id', 'customer_segment'],
    date_col: str= 'date',
    lag: int = 1,
    new_col_name: str | None = None
):
    df = df.copy()

    if new_col_name is None:
        new_col_name = f"{value_col}_l_{lag}"

    df = df.sort_values(group_col + [date_col])
    df[new_col_name] = (
        df
        .groupby(group_col, observed=True)[value_col]
        .shift(lag)
    )
    return df




def transformation(
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        eval_start:str,
        eval_end:str
        ):
    df = add_lag_feature(df, 'sell_price', lag=1)
    #df = add_lag_feature(df, 'sell_price', lag=365)
    df = add_lag_feature(df, 'sold', lag=1)

    train, eval = ts_data_split(
        df, start_date, end_date, eval_start, eval_end
    )
    cat_cols = ['dept_id', 'customer_segment']
    encoder = fit_customer_segment_ohe(train,cat_cols)
    train, cols = customer_segment_ohe(train, encoder, cat_cols)
    eval, _ = customer_segment_ohe(eval, encoder, cat_cols)

    return train, eval, cols