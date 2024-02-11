import pytest
from pydantic import ValidationError

from pyvolutionary import ContinuousVariable, ContinuousMultiVariable


def test_invalid_bounds_for_continuous_variable():
    with pytest.raises(ValidationError):
        ContinuousVariable(name="x1", lower_bound=-10.0, upper_bound=-11.0)


def test_invalid_bounds_for_continuous_multivariable():
    with pytest.raises(ValidationError):
        ContinuousMultiVariable(name="x1", lower_bounds=[-10.0, -10], upper_bounds=[10, -11.0])
