from pydantic import field_validator

from ..models import Agent, BaseOptimizationConfig


class Electromagnet(Agent):
    pass


class ElectromagneticFieldOptimizationConfig(BaseOptimizationConfig):
    """
    Configuration class for the Electromagnetic Field Optimization algorithm.
        r_rate (float): [0.1, 0.6], like mutation parameter in GA but for one variable.\n
        ps_rate (float): [0.5, 0.95], like crossover parameter in GA.\n
        p_field (float): [0.05, 0.3], portion of population, positive field.\n
        n_field (float): [0.3, 0.7], portion of population, negative field.
    """
    r_rate: float
    ps_rate: float
    p_field: float
    n_field: float
    
    @field_validator("r_rate")
    def r_rate_validator(cls, v):
        if not 0.1 <= v <= 0.6:
            raise ValueError("r_rate must be between 0.1 and 0.6")
        return v
    
    @field_validator("ps_rate")
    def ps_rate_validator(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError("ps_rate must be between 0.5 and 0.95")
        return v
    
    @field_validator("p_field")
    def p_field_validator(cls, v):
        if not 0.05 <= v <= 0.3:
            raise ValueError("p_field must be between 0.05 and 0.3")
        return v
    
    @field_validator("n_field")
    def n_field_validator(cls, v):
        if not 0.3 <= v <= 0.7:
            raise ValueError("n_field must be between 0.3 and 0.7")
        return v
