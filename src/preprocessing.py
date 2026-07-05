import pandas as pd


def convert_date(df):
    """Convert the Date column to datetime for proper time series handling."""
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def rename_columns(df):
    """Rename raw Yahoo Finance columns to simple, consistent names.

    Original format: Gold_('Close', 'GC=F') -> gold_close
    """
    df = df.copy()

    df = df.rename(columns={
        "Gold_('Close', 'GC=F')": "gold_close",
        "Gold_('High', 'GC=F')": "gold_high",
        "Gold_('Low', 'GC=F')": "gold_low",
        "Gold_('Open', 'GC=F')": "gold_open",
        "Gold_('Volume', 'GC=F')": "gold_volume",

        "DXY_('Close', 'DX-Y.NYB')": "dxy_close",
        "DXY_('High', 'DX-Y.NYB')": "dxy_high",
        "DXY_('Low', 'DX-Y.NYB')": "dxy_low",
        "DXY_('Open', 'DX-Y.NYB')": "dxy_open",
        "DXY_('Volume', 'DX-Y.NYB')": "dxy_volume",

        "VIX_('Close', '^VIX')": "vix_close",
        "VIX_('High', '^VIX')": "vix_high",
        "VIX_('Low', '^VIX')": "vix_low",
        "VIX_('Open', '^VIX')": "vix_open",
        "VIX_('Volume', '^VIX')": "vix_volume",

        "TNX_('Close', '^TNX')": "tnx_close",
        "TNX_('High', '^TNX')": "tnx_high",
        "TNX_('Low', '^TNX')": "tnx_low",
        "TNX_('Open', '^TNX')": "tnx_open",
        "TNX_('Volume', '^TNX')": "tnx_volume",
    })

    return df


def drop_uninformative_columns(df):
    """Drop volume columns that are not used as predictors."""
    df = df.copy()

    columns_to_drop = [
        "dxy_volume",
        "vix_volume",
        "tnx_volume",
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def remove_missing_values(df):
    """Drop rows with null values (few in this dataset, imputation not needed)."""
    df = df.copy()

    df = df.dropna()

    return df


def clean_gold_data(input_path, output_path):
    """Run full data cleaning pipeline and save to CSV.

    Steps: load raw CSV, convert dates, rename columns, drop unused columns, remove nulls.
    Output serves as input for feature_engineering.py.
    """
    df = pd.read_csv(input_path)

    df = convert_date(df)
    df = rename_columns(df)
    df = drop_uninformative_columns(df)
    df = remove_missing_values(df)

    df.to_csv(output_path, index=False)

    print(f"Dataset limpio guardado en {output_path} | shape: {df.shape}")

    return df


if __name__ == "__main__":
    clean_gold_data(
        "data/raw/gold-macro-data.csv",
        "data/processed/gold-clean.csv"
    )
