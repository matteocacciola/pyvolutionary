# Changelog

## 2.7.0

### Behaviour changes

These fixes change the results of some algorithms, even with the same seed.

- `PermutationVariable` spans one dimension per item ("random keys"): the position of an agent holds a key per item,
  and the permutation is the order of the items by increasing key (`transform_solution` returns it, as before). It was a
  single dimension holding a nested list, which 15 algorithms could not handle (e.g. Ant Colony, Bacterial Foraging,
  Battle Royale, Coral Reef, Egret Swarm, Energy Valley, Firefly Swarm, Forensic-Based Investigation, Forest, Golden
  Jackal, QLE Sine Cosine): all the algorithms now solve permutation tasks. The corrected position holds the ranks of the
  keys, so that correcting it again leaves it as it is (it was sorted twice before evaluating the objective).

- `Task.increase_solution` now moves from the given solution, instead of returning a random one. This changes Osprey,
  Walrus, Siberian Tiger and Wildebeest Herd optimizations.
- The velocity of Particle Swarm Optimization is now clamped to 20% of the search range, as intended.
- The Lévy flight step (`get_levy_flight_step`) uses `sigma_u` as the standard deviation of `u` (it used the variance).
  This changes all the algorithms based on Lévy flights (e.g. Cuckoo Search, Flower Pollination, Marine Predators,
  Golden Jackal, Aquila, Dragonfly, Monarch Butterfly).
- Water Cycle Optimization sorts the initial population, so that the best agent is the sea and the following ones are
  the rivers; every river gets at least one stream.
- Firefly Swarm Optimization follows the implementation of mealpy (`OriginalFFA`). Before:
  - the mutation coefficient `alpha` collapsed at the first cycle (to ~1e-4 of its value, and to ~1e-12 at the tenth),
    since its decay used the current cycle instead of the number of cycles, and it was squared at every cycle;
  - a firefly was always replaced by its best candidate, even when worse, so good solutions were lost;
  - each move started from the original position, instead of the one reached by the previous move;
  - one random candidate more than the population size was generated.

  Now `alpha` is damped by `alpha_damp` at every cycle, a firefly is replaced only by a better candidate, and the
  attraction is `beta_min * exp(-gamma * r^exponent)` with a random step scaled by `delta`. The new `alpha_damp`
  (0.99), `delta` (0.05) and `exponent` (2) parameters have mealpy's defaults, and `beta_min` (the base attraction,
  `beta_base` in mealpy) accepts values up to 3. With the same parameters, the results are as good as mealpy's (median
  best cost over 10 seeds on a 10-dimensional Sphere: 50.6 vs 48.4 of mealpy and 102 before; on Rastrigin: 71.7 vs 71.1
  and 93.2 before). Two mealpy bugs are not reproduced: mealpy damps the initial `alpha` (so it stays constant after the
  first cycle), and compares the best candidate with the firefly after its moves instead of the firefly itself.
- Imperialist Competitive Optimization:
  - the initial countries are unique, as intended (the check never discarded a duplicate);
  - the revolution exchanges the candidate dimensions with the other ones (it removed the wrong indexes from the
    list, and failed with a single dimension), and works on a copy of the colony, which is no longer altered when the
    new colony is discarded (its cost did not match its representation anymore);
  - the power of the empires is computed by scaling the costs by their largest magnitude: with negative costs (e.g.
    of maximization tasks) the weakest empires were the most likely to win, and the weights could overflow. Nothing
    changes with positive costs.
- Earthworms Optimization re-samples only the actual duplicates of the population: each earthworm was considered a
  duplicate of itself, so almost the whole population was altered and evaluated again at every cycle. The dimension to
  re-sample can be any one (the last one was never chosen, and a single dimension failed).
- Bee Colony and Biogeography-Based optimizations scale the costs by the magnitude of their sum: with negative costs
  (e.g. of maximization tasks) the roulette wheel favoured the worst agents, and with all costs equal to zero it failed.
  Nothing changes with positive costs.
- Early stopping now stops when the error has not improved by at least `min_delta` for `patience` cycles. Before, it
  never stopped when the error stagnated or got worse.
- `HyperTuner` ranks the standard deviation (lower is better) and the combined rank correctly for maximization tasks.

### Fixes

- `optimize()` can be called several times on the same instance: the cycle counter and the error history are reset.
- Coral Reef, Fox, Imperialist Competitive and Success History Intelligent optimizations reset their state at every
  run: optimizing again with the same instance gave different results (Imperialist Competitive kept the empires of the
  previous run). Coral Reef can be created without a configuration, as the other algorithms.
- African Vulture and Fox optimizations no longer produce NaN positions (a division by zero, and `inf * 0` at the first
  cycle), which failed with discrete variables; Ficks Law no longer fails when a cost is zero.
- Bee Colony and Firefly Swarm optimizations no longer alter their configuration. Bee Colony halved
  `population_size` at every run, so a reused configuration ended up with an empty colony (or hung with a single bee).
- In `process` mode, each worker is seeded independently: workers generated identical agents.
- Results of parallel jobs keep the order of submission.
- Multi-objective maximization tasks no longer fail.
- `Task.amend_solution` no longer fails on array comparisons.
- `random_selection` no longer raises `IndexError` on rounding errors.
- `best_agent_formatted` and `worst_agent_formatted` no longer alter the agents of the population.
- `DiscreteMultiVariable` returns its bounds as `(lower bounds, upper bounds)`, as the other variables: a task got
  wrong bounds with two discrete variables, and failed with any other number of them.
- `transform_solution` decodes a multi-variable of one dimension as a list (it failed).
- A `Task` follows its variables when they are replaced (`task.variables = ...` or
  `task.model_copy(update={"variables": ...})`): `space_dimension` and the bounds were left as they were.
- Energy Valley, Forest and Golden Jackal optimizations no longer fail (or truncate the updates) with integer positions,
  e.g. of discrete or permutation variables.
- `distances` returns the full matrix of pairwise distances (it dropped the last coordinate).
- `Multitask` accepts one mode per algorithm, and repeated `execute()` calls no longer accumulate results.
- `HyperTuner` and `Multitask` work on machines with 2 CPUs or less.
- The `random` module is seeded together with numpy.

### Performance

- Solutions are corrected with a single vectorized operation when all the variables are continuous, and the variables
  and bounds of a task are cached: optimizations are 2-5x faster.
- Random solutions are drawn with a single vectorized operation when all the variables are continuous, and the roulette
  wheel selection no longer builds Python lists at every call.
- The implementations of the algorithms are optimized without changing their results: the evolution of every algorithm
  is bit-identical to the previous version for the same seed. Among the largest speedups: Ant Colony 7x, Grasshopper
  6.6x, Krill Herd 3.6x, Genetic Algorithm 2.8x, Dragonfly 2.6x, Ant Lion 2.5x, Fireworks 2.2x, Imperialist
  Competitive 1.9x. Krill Herd evaluates the food position once per cycle instead of once per krill.
- Ant Lion and Fireworks no longer fail on permutation tasks.
- `HyperTuner` and `Multitask` share a single pool of processes.
- Greedy selection no longer dispatches trivial work to a pool of workers.
