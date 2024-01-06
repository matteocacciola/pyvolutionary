from .functions.ackley import Ackley
from .functions.elliptic import Elliptic
from .functions.griewank import Griewank
from .functions.michalewicz import Michalewicz
from .functions.rastrigin import Rastrigin
from .functions.rosenbrock import Rosenbrock
from .functions.sphere import Sphere
from .functions.weierstrass import Weierstrass
from .functions.zakharov import Zakharov
from pyvolutionary import plot as py_plot, animate as py_animate


def plot(**kwargs):
    py_plot(**kwargs)


def animate(**kwargs):
    py_animate(**kwargs)