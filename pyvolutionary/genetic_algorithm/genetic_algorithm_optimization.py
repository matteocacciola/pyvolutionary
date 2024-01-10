import numpy as np

from ..helpers import parse_obj_doc  # type: ignore
from ..abstract import OptimizationAbstract
from .classes import BitGene, Transformer
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
        self.__p_mutation: float | None = None
        self.__bit_genes: list[BitGene] | None = None

    def before_initialization(self):
        self.__bit_genes = [BitGene(self._task.space_dimension) for _ in range(0, self._config.population_size)]
        self.__p_mutation = 1.0 / (self.__bit_genes[0].n_bits * self._task.space_dimension)

    def _init_population(self):
        lb, ub = self._get_bounds()
        self._population = [Transformer.from_bit_gene_to_gene(
            bit_gene, lb.tolist(), ub.tolist(), self._init_agent
        ) for bit_gene in self.__bit_genes]

    def optimization_step(self):
        def selection(k: int | None = 3) -> list[int]:
            # first random selection
            selection_ix = np.random.randint(len(self.__bit_genes))
            for ix in np.random.randint(0, len(self.__bit_genes), k - 1):
                # check if better (e.g. perform a tournament)
                if scores[ix] < scores[selection_ix]:
                    selection_ix = ix
            return self.__bit_genes[selection_ix].bitstring

        def crossover(idx: int) -> tuple[list[int], list[int]]:
            # get selected parents in pairs
            parent1, parent2 = selected[idx].copy(), selected[idx + 1].copy()
            # children are copies of parents by default
            child1, child2 = parent1.copy(), parent2.copy()
            # check for recombination
            if np.random.rand() < px_over:
                # select crossover point that is not on the end of the string
                pt = np.random.randint(1, len(parent1) - 2)
                # perform crossover
                child1 = parent1[:pt] + parent2[pt:]
                child2 = parent1[:pt] + parent2[pt:]
            return child1, child2

        px_over = self._config.px_over
        scores = [gene.cost for gene in self._population]

        # select parents
        selected = [selection() for _ in range(0, self._config.population_size)]

        # create the next generation
        children = []
        for i in range(0, self._config.population_size, 2):
            # crossover
            c1, c2 = crossover(i)
            # mutations and store for next generation
            children.append([1 - bit if np.random.rand() < self.__p_mutation else bit for bit in c1])
            children.append([1 - bit if np.random.rand() < self.__p_mutation else bit for bit in c2])

        # update population
        self.__bit_genes = [self.__bit_genes[idx].set_bit_string(child) for idx, child in enumerate(children)]

        lb, ub = self._get_bounds()
        self._population = [Transformer.from_bit_gene_to_gene(
            bit_gene, lb.tolist(), ub.tolist(), self._init_agent
        ) for bit_gene in self.__bit_genes]
