import pytest
import numpy as np
from flicker import is_nan_scalar


def test_int():
    assert is_nan_scalar(1) is False


def test_non_nan_float():
    assert is_nan_scalar(1.5435) is False


def test_nan():
    assert is_nan_scalar(np.nan) is True


def test_string():
    assert is_nan_scalar('42') is False
    assert is_nan_scalar('np.nan') is False
    assert is_nan_scalar('nan') is False
    assert is_nan_scalar('NaN') is False
    assert is_nan_scalar('hello') is False


def test_non_scalar():
    with pytest.raises(ValueError):
        is_nan_scalar([1, 2, 3])
    with pytest.raises(ValueError):
        is_nan_scalar([1, np.nan, 3.0])


def test_boolean_scalar():
    assert is_nan_scalar(True) is False
    assert is_nan_scalar(False) is False
    assert is_nan_scalar(None) is False
