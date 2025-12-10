# MarxSim

Reinforcement learning in a Marx-inspired agent-based wealth distribution environment.

This project implements:

- A **MarxSim environment** with workers vs capitalists, unequal exchanges, group organization, and revolutions.
- A **Deep Q-Network (DQN)** agent that learns how to act in this environment using standard RL techniques.

The idea is to explore how an RL agent behaves in a stylized Marxian economy with inequality, class struggle, and regime change.

---

## 1. Environment Overview: `MarxSimEnv`

The environment is implemented in `MarxSimEnv` and is designed to be RL-ready with `reset()` and `step(action)`.

### Agents and Classes

- `num_agents`: total number of agents (default `200`)
- `frac_capitalists`: fraction of agents that are capitalists (default `0.1`)
  - Capitalists: benefit from a **biased wealth exchange** mechanism.
  - Workers: majority class, divided into groups.

### Worker Groups & Unity

- Workers are assigned to `num_groups` groups (default `3`).
- Each group has a **unity level** in `[0, 1]`:
  - Higher unity → more chance of successful revolt.
  - Unity can be increased (at a cost) or damaged (failed revolts, leaving the group).

### Regimes

- **Capitalist regime** (default at start):
  - Wealth exchanges are **biased** in favor of capitalists using `power_bias`.
- **Post-capitalist regime** (after a successful revolution):
  - Capitalist privilege is removed.
  - Wealth moves softly toward equality via equalizing dynamics.

### Step Function and Actions

`step(action_index)`:

- `action_index` is decoded as:
  - `group = action_index // 4`
  - `action_type = action_index % 4`
- For each worker group, there are 4 possible actions:
  - `0`: Do nothing
  - `1`: **Unify** – increase group unity (costs some wealth)
  - `2`: **Revolt** – attempt a revolution based on unity and inequality
  - `3`: **Leave** – some workers leave the group, reducing unity

The environment then:

1. Applies the chosen group action.
2. Applies wealth dynamics:
   - Capitalist regime → biased exchanges.
   - Post-capitalist regime → equalizing dynamics.
3. Computes stats and a scalar reward.
4. Returns `(state, reward, done, info)`.

### State Representation

The state is a vector of 6 features:

1. `gini` – Gini coefficient of wealth.
2. `top10` – share of total wealth held by the top 10% richest agents.
3. `mean_worker` – average wealth of workers.
4. `mean_cap` – average wealth of capitalists.
5. `avg_unity` – average group unity across all groups.
6. `max_unity` – maximum unity among all groups.

This is returned as a NumPy array of shape `(6,)` with `dtype=float32`.

### Reward Function

The reward combines:

- Change in workers’ average wealth.
- Change in inequality (Gini coefficient).
- Bonuses/penalties for political actions:
  - `+ lambda_succ` for a **successful revolution**
  - `- lambda_fail` for a **failed revolution**
  - Small penalties for unify/leave actions to encode their costs.

The lambdas (`lambda_w`, `lambda_g`, `lambda_succ`, etc.) are defined in the environment and can be tuned.

---

## 2. DQN Agent

The RL agent is a standard Deep Q-Network (DQN):

### Network Architecture

Defined in the `DQN` class:

- Input: `state_dim = 6`
- Two hidden layers of size 64 with ReLU activations:
  - `Linear(6 → 64) → ReLU`
  - `Linear(64 → 64) → ReLU`
  - `Linear(64 → action_dim)`
- Output: Q-values for each discrete action.

### DQN Details

Implemented in `DQNAgent`:

- Experience replay buffer using `collections.deque`.
- Target network with periodic updates (`target_update_freq`).
- Epsilon-greedy exploration:
  - Starts at `epsilon_start = 1.0`
  - Decays to `epsilon_end = 0.05`
  - Decay rate `epsilon_decay = 0.995`
- Optimizer: `Adam` with learning rate `1e-3`.
- Discount factor: `gamma = 0.99`.

---

## 3. Training Loop

Training is handled by the `train_dqn` function:

```python
agent, train_returns, eval_data = train_dqn(
    num_episodes=5000,
    eval_every=20,
    seed=123
)
