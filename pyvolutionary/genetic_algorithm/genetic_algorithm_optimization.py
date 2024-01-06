import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .models import Gene, GeneticAlgorithmOptimizationConfig


class GeneticAlgorithmOptimization(OptimizationAbstract):
    """
    Implementation of the Genetic Algorithm Optimization algorithm.

    Args:
        config (GeneticAlgorithmOptimizationConfig): an instance of GeneticAlgorithmOptimizationConfig class.
            {parse_obj_doc(GeneticAlgorithmOptimizationConfig)}

    Bibliography
    ----------
    [1] Alander, Jarmo. (1999). An Indexed Bibliography of Genetic Algorithm Implementations.
    [2] Goldberg, David E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
        ISBN 978-0201157673. OCLC 18319123.
    [3] Holland, John H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.
        ISBN 978-0262581110. OCLC 1635804.
    """
    def __init__(self, config: GeneticAlgorithmOptimizationConfig, debug: bool | None = False):
        super().__init__(config, debug)
        self.__n_bits: int | None = None
        self.__p_mutation: float | None = None
        self.__bit_positions: list[list[int]] | None = None

    def before_initialization(self):
        self.__n_bits = 8 * self._task.space_dimension
        self.__p_mutation = 1.0 / (float(self.__n_bits) * self._task.space_dimension)

    def _init_population(self):
        self.__bit_positions = [np.random.randint(
            0, 2, self.__n_bits * self._task.space_dimension
        ).tolist() for _ in range(0, self._config.population_size)]

        self._population = self.__from_bit_positions_to_population__()

    def __decode__(self, bitstring: list[int]) -> list[float]:
        """
        Decode a bitstring to a float list.
        :param bitstring: the bitstring to decode
        :return: the decoded float list
        :rtype: list[float]
        """
        decoded = []
        largest = 2 ** self.__n_bits

        lb, ub = self._get_bounds()
        for i in range(0, self._task.space_dimension):
            # extract the substring
            start, end = i * self.__n_bits, (i * self.__n_bits) + self.__n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = lb[i] + (integer / largest) * (ub[i] - lb[i])
            # store
            decoded.append(value)
        return decoded

    def __encode__(self, position: list[float]) -> list[int]:
        """
        Encode a float list to a bitstring.
        :param position: the float list to encode
        :return: the encoded bitstring
        :rtype: list[int]
        """
        encoded = []

        lb, ub = self._get_bounds()
        for i in range(0, self._task.space_dimension):
            # scale integer to desired range
            value = (position[i] - lb[i]) / (ub[i] - lb[i])
            # convert to bitstring
            value = bin(int(value * (2 ** self.__n_bits)))[2:].zfill(self.__n_bits)
            binary_representation = [int(bit) for bit in value]
            # append to the encoded list
            encoded.extend(binary_representation)
        return encoded

    def __selection__(self, k: int | None = 3) -> list[int]:
        """
        Select a parent from the population. The selection is based on the fitness of the genes. The lower the cost,
        the higher the fitness. The selection is performed by tournament selection. The tournament size is k. The
        selection is repeated until k parents are selected. The best parent is selected from the k parents. The best is
        the one with the lowest cost. The selection is repeated until the population size is reached. The selection is
        performed with replacement. The same parent can be selected multiple times. The selection is performed with a 3
        tournament selection by default.
        :param k: the tournament size
        :return: the selected parent
        """
        scores = [gene.cost for gene in self._population]

        # first random selection
        selection_ix = np.random.randint(len(self.__bit_positions))
        for ix in np.random.randint(0, len(self.__bit_positions), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.__bit_positions[selection_ix]

    def __crossover__(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Crossover two parents to create two children. The crossover is performed with a probability of px_over. The
        crossover point is selected randomly. The crossover point is not on the end of the string. The crossover is
        performed by default. The children are copies of the parents by default.
        :param parent1: the first parent
        :param parent2: the second parent
        :return: the two children
        :rtype: tuple[list[int], list[int]]
        """
        # children are copies of parents by default
        c1, c2 = parent1.copy(), parent2.copy()
        # check for recombination
        if np.random.rand() < self._config.px_over:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(parent1) - 2)
            # perform crossover
            c1 = parent1[:pt] + parent2[pt:]
            c2 = parent1[:pt] + parent2[pt:]
        return c1, c2

    def __mutation__(self, bitstring: list[int]) -> list[int]:
        """
        Mutate a bitstring by flipping bits with a probability of pm_mutate. The mutation is performed by default.
        :param bitstring: the bitstring to mutate
        :return: the mutated bitstring
        :rtype: list[int]
        """
        return [1 - bit if np.random.rand() < self.__p_mutation else bit for bit in bitstring]

    def __from_bit_positions_to_population__(self) -> list[Gene]:
        """
        Convert the bit positions to a population. The cost of each gene is calculated. The fitness of each gene is
        calculated.
        :return: the population
        :rtype: list[Gene]
        """
        def to_gene(bitstring: list[int]) -> Gene:
            position = self.__decode__(bitstring)
            return Gene(**self._init_agent(position).model_dump())

        return list(map(to_gene, self.__bit_positions))

    def optimization_step(self):
        # select parents
        selected = [self.__selection__() for _ in range(0, self._config.population_size)]

        # create the next generation
        children = []
        for i in range(0, self._config.population_size, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover
            c1, c2 = self.__crossover__(p1, p2)
            # mutations and store for next generation
            children.append(self.__mutation__(c1))
            children.append(self.__mutation__(c2))

        # update population
        self.__bit_positions = children
        self._population = self.__from_bit_positions_to_population__()
