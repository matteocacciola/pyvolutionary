import pytest
from pydantic import ValidationError

from pyvolutionary import ContinuousVariable


def test_invalid_bounds():
    with pytest.raises(ValidationError):
        ContinuousVariable(name="x1", lower_bound=-10.0, upper_bound=-11.0)
