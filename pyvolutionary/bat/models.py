from pydantic import field_validator, conlist

from ..models import Agent, BaseOptimizationConfig


class Bat(Agent):
    velocity: list[float]
    loudness: float
    pulse_rate: float


class BatOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for Bat Optimization algorithm.
        loudness (list[float]): [0.5, 1.5], [1.5, 3.0], loudness range.\n
        pulse_rate (list[float]): (0, 1.0), (0, 1.0), pulse rate range.\n
        pulse_frequency (list[float]): [-10, 0], [0, 10], pulse frequency range (both cannot be 0).\n
        alpha (float): [0, 1], loudness update parameter.\n
        gamma (float): [0, 1], pulse rate update parameter.
    """
    loudness: conlist(float, min_length=2, max_length=2)
    pulse_rate: conlist(float, min_length=2, max_length=2)
    pulse_frequency: conlist(float, min_length=2, max_length=2)
    alpha: float = 0.9
    gamma: float = 0.9

    @field_validator("loudness")
    def correct_loudness(cls, v):
        loudness_min, loudness_max = v
        if not 0.5 <= loudness_min <= 1.5:
            raise ValueError(f"\"loudness[0]\" must be a float in [0.5, 1.5]. Got {loudness_min}")
        if not 1.5 <= loudness_max <= 3.0:
            raise ValueError(f"\"loudness[1]\" must be a float in [1.5, 3.0]. Got {loudness_max}")
        return v

    @field_validator("pulse_rate")
    def correct_pulse_rate(cls, v):
        pulse_rate_min, pulse_rate_max = v
        if not 0 < pulse_rate_min < 1:
            raise ValueError(f"\"pulse_rate[0]\" must be a float in (0, 1.0). Got {pulse_rate_min}")
        if not 0 < pulse_rate_max < 1:
            raise ValueError(f"\"pulse_rate[1]\" must be a float in (0, 1.0). Got {pulse_rate_max}")
        return v

    @field_validator("pulse_frequency")
    def correct_pulse_frequency(cls, v):
        pulse_frequency_min, pulse_frequency_max = v
        if not -10 <= pulse_frequency_min <= 0:
            raise ValueError(f"\"pulse_frequency[0]\" must be a float in [-10, 0]. Got {pulse_frequency_min}")
        if not 0 <= pulse_frequency_max <= 10:
            raise ValueError(f"\"pulse_frequency[1]\" must be a float in [0, 10]. Got {pulse_frequency_max}")
        if pulse_frequency_max == 0 or pulse_frequency_max == 0:
            raise ValueError(f"\"pulse_frequency\" cannot have both 0 values. Got {v}")
        return v

    @field_validator("alpha")
    def correct_alpha(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"\"alpha\" must be a float in [0, 1]. Got {v}")
        return v

    @field_validator("gamma")
    def correct_gamma(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"\"gamma\" must be a float in [0, 1]. Got {v}")
        return v
