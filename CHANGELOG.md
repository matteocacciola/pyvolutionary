# Changelog

## 2.7.0

### Behaviour changes

These fixes change the results of some algorithms, even with the same seed.

- `Task.increase_solution` now moves from the given solution, instead of returning a random one. This changes Osprey,
  Walrus, Siberian Tiger and Wildebeest Herd optimizations.
- The velocity of Particle Swarm Optimization is now clamped to 20% of the search range, as intended.
- The Lévy flight step (`get_levy_flight_step`) uses `sigma_u` as the standard deviation of `u` (it used the variance).
  This changes all the algorithms based on Lévy flights (e.g. Cuckoo Search, Flower Pollination, Marine Predators,
  Golden Jackal, Aquila, Dragonfly, Monarch Butterfly).
- Water Cycle Optimization sorts the initial population, so that the best agent is the sea and the following ones are
  the rivers; every river gets at least one stream.
- Firefly Swarm Optimization decays `alpha` as `alpha = (1 - delta) * alpha`: it squared `alpha` at every cycle.
- Early stopping now stops when the error has not improved by at least `min_delta` for `patience` cycles. Before, it
  never stopped when the error stagnated or got worse.
- `HyperTuner` ranks the standard deviation (lower is better) and the combined rank correctly for maximization tasks.

### Fixes

- `optimize()` can be called several times on the same instance: the cycle counter and the error history are reset.
- Bee Colony and Firefly Swarm optimizations no longer alter their configuration. Bee Colony halved
  `population_size` at every run, so a reused configuration ended up with an empty colony (or hung with a single bee).
- In `process` mode, each worker is seeded independently: workers generated identical agents.
- Results of parallel jobs keep the order of submission.
- Multi-objective maximization tasks no longer fail.
- `Task.amend_solution` no longer fails on array comparisons.
- `random_selection` no longer raises `IndexError` on rounding errors.
- `best_agent_formatted` and `worst_agent_formatted` no longer alter the agents of the population.
- `distances` returns the full matrix of pairwise distances (it dropped the last coordinate).
- `Multitask` accepts one mode per algorithm, and repeated `execute()` calls no longer accumulate results.
- `HyperTuner` and `Multitask` work on machines with 2 CPUs or less.
- The `random` module is seeded together with numpy.

### Performance

- Solutions are corrected with a single vectorized operation when all the variables are continuous, and the variables
  and bounds of a task are cached: optimizations are 2-5x faster.
- Random solutions are drawn with a single vectorized operation when all the variables are continuous, and the roulette
  wheel selection no longer builds Python lists at every call.
- `HyperTuner` and `Multitask` share a single pool of processes.
- Greedy selection no longer dispatches trivial work to a pool of workers.
