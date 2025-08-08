import os
import numpy as np
import pandas as pd
from utils import LogDiffMinMaxScaler, seed_everything


def test_scaler_method_selection():
    # Positive-only series -> 'log'; zero/negative -> 'diff'
    df = pd.DataFrame({
        'Date': pd.date_range('2021-01-01', periods=10, freq='h').astype(str),
        'Open': np.linspace(100.0, 110.0, 10),            # positive -> log
        'High': np.linspace(120.0, 130.0, 10),            # positive -> log
        'Low':  np.linspace(90.0, 95.0, 10),              # positive -> log
        'Close':np.linspace(100.0, 105.0, 10),            # positive -> log
        'Volume': np.array([0,1,2,3,4,5,6,7,8,9], float), # includes zero -> diff
        'CustomNeg': np.linspace(-1.0, 1.0, 10),          # negatives -> diff
    })
    scaler = LogDiffMinMaxScaler().fit(df)
    assert scaler.column_to_method['Open'] == 'log'
    assert scaler.column_to_method['High'] == 'log'
    assert scaler.column_to_method['Low'] == 'log'
    assert scaler.column_to_method['Close'] == 'log'
    assert scaler.column_to_method['Volume'] == 'diff'
    assert scaler.column_to_method['CustomNeg'] == 'diff'

    # Transform should produce finite values except first row due to shift
    df_t = scaler.transform(df)
    assert np.isfinite(df_t.iloc[1:,1:].to_numpy()).all()


def test_seed_everything_sets_env():
    seed_everything(999)
    assert os.environ.get('PYTHONHASHSEED') == '999'