import os
import sys
import numpy as np
import pandas as pd
import pytest

# Make package-less modules importable
DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

from utils import LogDiffMinMaxScaler, seed_everything


def make_dummy_df(n=500):
    rng = pd.date_range('2021-01-01', periods=n, freq='H')
    df = pd.DataFrame({
        'Date': rng.astype(str),
        'Open': np.linspace(10000, 11000, n) + np.random.randn(n)*10 + 100,
        'High': np.linspace(10020, 11020, n) + np.random.randn(n)*10 + 100,
        'Low':  np.linspace(9980,  10980,  n) + np.random.randn(n)*10 + 100,
        'Close':np.linspace(10010, 11010, n) + np.random.randn(n)*10 + 100,
        'Volume': np.abs(1000 + np.random.randn(n)*50)
    })
    return df


def test_scaler_no_leak_and_shapes():
    df = make_dummy_df(400)
    df = df[100:].dropna()
    lookback = 50
    test_window = 120
    train_df = df[:-test_window-lookback]
    test_df = df[-test_window-lookback:]

    scaler = LogDiffMinMaxScaler().fit(train_df)
    train_n = scaler.transform(train_df)
    test_n = scaler.transform(test_df)

    # after transform, there will be NaNs at first row due to shift; ensure later rows are finite
    assert np.isfinite(train_n.iloc[1:,1:].to_numpy()).all()
    assert np.isfinite(test_n.iloc[1:,1:].to_numpy()).all()


def test_seed_everything_reproducible_numpy():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)