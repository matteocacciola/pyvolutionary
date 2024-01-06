from pydantic import field_validator, model_validator, conlist

from ..models import Agent, BaseOptimizationConfig


class InvasiveWeed(Agent):
    pass


class InvasiveWeedOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class of the Invasive Weed Optimization algorithm.
        seed (list[int]): [1, 3], [4, pop_size/2], minmax numbers of Seeds.\n
        exponent (int): [2, 4], variance reduction exponent.\n
        sigma (list[float]): [0.5, 5.0], (0, 0.5), the initial minmax values of Standard Deviation.
    """
    seed: conlist(int, min_length=2, max_length=2)
    exponent: int
    sigma: conlist(float, min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_seed(self) -> "InvasiveWeedOptimizationConfig":
        seed_min, seed_max = self.seed
        if seed_max > self.population_size / 2:
            raise ValueError(f"\"seed_max\" must be less than the half of the population_size. Got {seed_max}")
        return self

    @field_validator("seed")
    def correct_seed(cls, v):
        seed_min, seed_max = v

        if not 1 <= seed_min <= 3:
            raise ValueError(f"\"seed[0]\" must be a positive int within the range [1, 3]. Got {seed_min}")
        if seed_max < 4:
            raise ValueError(f"\"seed[1]\" must be a positive int within the range [4, pop_size/2]. Got {seed_max}")
        return v

    @field_validator("exponent")
    def correct_exponent(cls, v):
        if not 2 <= v <= 4:
            raise ValueError(f"\"exponent\" must be a positive int within the range [2, 4]. Got {v}")
        return v

    @field_validator("sigma")
    def correct_sigma(cls, v):
        start, end = v
        if not 0.5 <= start <= 5:
            raise ValueError(f"\"sigma[0]\" must be a positive float within the range [0.5, 5]. Got {start}")
        if not 0 < end < 0.5:
            raise ValueError(f"\"sigma[1]\" must be a positive float within the range (0, 0.5). Got {end}")
        return v
