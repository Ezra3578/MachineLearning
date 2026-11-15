# MDP Specification — Workshop 3 (RL on Fashion-MNIST)

> Code artifact for item 1 of the guide (`MDP Formulation Document`).

## 1. Problem Statement

The RL agent solves Fashion-MNIST classification (10 classes) as a sequential
decision process: at each step it observes one image and chooses the predicted
label. RL is used as an alternative/complementary approach to the supervised
models of Workshop 1 (Random Forest / XGBoost / HGB) and Workshop 2
(SmallCNN / ResNet-18).

RL is appropriate here because: (a) the task can be formulated as sequential
decisions where every prediction receives an immediate, verifiable reward;
(b) we have a cheap simulator (i.i.d. sampling from the dataset); (c) it exposes
exploration vs. exploitation, reward shaping and sample-efficiency against
supervised learning. We honestly report that the environment is a contextual
bandit (transition independent of the action), so supervised learning will be
more sample-efficient: the comparison is the scientific point of the workshop.

## 2. State Space

- Observation: 28x28 grayscale Fashion-MNIST image, normalized with the global
  Workshop 1/2 statistics (`mean = 0.2860`, `std = 0.3530`).
- Network representation: `float32` tensor of shape `(1, 28, 28)`. Pixels in
  [0,255] are scaled to [0,1] and then standardized with the global statistics.
- `observation_space = Box(low=-5, high=5, shape=(1, 28, 28), dtype=float32)`
  (safely contains the standardized values; see `src/env.py`).
- No temporal stacking and no history: the state is Markovian by construction
  (i.i.d. sampling), `state_dim = 784` effective pixels.
- Split used: the same stratified 70/15/15 split with `seed=42` from Workshop 2,
  so metrics stay comparable.

## 3. Action Set

- Discrete, `action_space = Discrete(10)`, enumeration identical to
  `CLASS_NAMES`:
  `0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt,`
  `7 Sneaker, 8 Bag, 9 Ankle boot`.
- Interpretation: action `a_t` IS the predicted label for `s_t`.
- Being discrete and small justifies DQN (guide, Agent Algorithm section).

## 4. Transition Dynamics

- `s_{t+1} = f(s_t, a_t) = sample_uniform(split)`: the next image is sampled
  uniformly from the active split, regardless of the action taken.
- Deterministic given the sampled index; stochastic from the agent's
  perspective (i.i.d.). No latent dynamics and no memory between steps.
- Consequence: `gamma` has a limited effect (immediate reward dominates); we
  still sweep `gamma in {0.90, 0.95, 0.99}` as the guide requests, to verify it
  empirically.

## 5. Reward Function

- Default (`reward_mode="plus_minus"`, dense and symmetric):
  `r_t = +1` if `a_t == y_t`, `-1` otherwise.
- Episode return with `N = max_steps = 100`:
  `G in [-100, 100]`; `accuracy = (G + N) / (2N)`.
- Alternative modes (`--reward-mode`): `zero_one` (`+1/0`, useful to compare
  against direct accuracy) and `sparse` (0 at intermediate steps,
  `+#correct` at the end; only for the guide's sparse-reward analysis, slow).
- Relevant horizon: the accumulated episode return (long term = sum of
  decisions); each step is also interpretable on its own (short term = hit/miss).

## 6. Episode Termination

- `truncated = True` when `t == max_steps` (default 100). `terminated` never
  fires (there is no absorbing success/failure state).
- Termination equals a "classification budget per episode" and keeps returns
  comparable across episodes. Task success is measured by mean return and
  accuracy, not by early termination.
