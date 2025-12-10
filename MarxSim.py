import numpy as np
import random
from collections import deque
import math

import torch
import torch.nn as nn
import torch.optim as optim


def gini(x):
    x = np.asarray(x, dtype=float)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


class MarxSimEnv:
    def __init__(
        self,
        num_agents=200,
        frac_capitalists=0.1,
        num_groups=3,
        init_wealth=100.0,
        power_bias=0.7,
        success_alpha=4.0,
        success_beta=4.0,
        max_steps=200,
        exchanges_per_step=200,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.num_agents = num_agents
        self.frac_capitalists = frac_capitalists
        self.num_groups = num_groups
        self.init_wealth = init_wealth
        self.base_power_bias = power_bias
        self.power_bias = power_bias
        self.success_alpha = success_alpha
        self.success_beta = success_beta
        self.max_steps = max_steps
        self.exchanges_per_step = exchanges_per_step

        self.lambda_w = 1.0
        self.lambda_g = 0.5
        self.lambda_succ = 3.0
        self.lambda_fail = 1.5
        self.lambda_u = 0.1
        self.lambda_l = 0.05

        self.unity_step = 0.05
        self.unity_decay_failed = 0.5

        self.t = 0
        self.wealth = None
        self.is_capitalist = None
        self.group_id = None
        self.group_unity = None
        self.regime = "capitalist"
        self.revolution_occurred = False

        self.prev_mean_worker = 0.0
        self.prev_gini = 0.0

    def reset(self):
        self.t = 0
        self.regime = "capitalist"
        self.revolution_occurred = False
        self.power_bias = self.base_power_bias

        self.wealth = np.full(self.num_agents, self.init_wealth, dtype=float)

        num_cap = int(self.frac_capitalists * self.num_agents)
        idx = np.arange(self.num_agents)
        np.random.shuffle(idx)
        cap_idx = idx[:num_cap]
        worker_idx = idx[num_cap:]

        self.is_capitalist = np.zeros(self.num_agents, dtype=bool)
        self.is_capitalist[cap_idx] = True

        self.group_id = np.full(self.num_agents, -1, dtype=int)
        if len(worker_idx) > 0:
            worker_groups = np.random.randint(0, self.num_groups, size=len(worker_idx))
            self.group_id[worker_idx] = worker_groups

        self.group_unity = np.full(self.num_groups, 0.2, dtype=float)

        stats = self._compute_stats()
        self.prev_mean_worker = stats["mean_worker"]
        self.prev_gini = stats["gini"]

        return self._state_from_stats(stats)

    def step(self, action_index):
        self.t += 1

        stats_before = self._compute_stats()

        group, action_type = self._decode_action(action_index)

        did_unify = False
        did_leave = False
        revolt_success = False
        revolt_fail = False

        if group is not None:
            if action_type == 1:
                did_unify = self._apply_unify(group)
            elif action_type == 2:
                revolt_success, revolt_fail = self._apply_revolt(group, stats_before)
            elif action_type == 3:
                did_leave = self._apply_leave(group)

        if self.regime == "capitalist":
            self._biased_exchanges()
        else:
            self._equalizing_dynamics()

        stats_after = self._compute_stats()

        reward = self._compute_reward(
            stats_before,
            stats_after,
            did_unify,
            did_leave,
            revolt_success,
            revolt_fail
        )

        done = self.t >= self.max_steps

        self.prev_mean_worker = stats_after["mean_worker"]
        self.prev_gini = stats_after["gini"]

        state = self._state_from_stats(stats_after)

        info = {
            "stats_before": stats_before,
            "stats_after": stats_after,
            "revolt_success": revolt_success,
            "revolt_fail": revolt_fail
        }

        return state, reward, done, info

    def _decode_action(self, action_index):
        total_actions = self.num_groups * 4
        if action_index < 0 or action_index >= total_actions:
            return None, 0
        g = action_index // 4
        a = action_index % 4
        return g, a

    def _worker_indices(self):
        return np.where(~self.is_capitalist)[0]

    def _capitalist_indices(self):
        return np.where(self.is_capitalist)[0]

    def _apply_unify(self, group):
        worker_idx = self._worker_indices()
        in_group = worker_idx[self.group_id[worker_idx] == group]
        if len(in_group) == 0:
            return False

        self.group_unity[group] = min(1.0, self.group_unity[group] + self.unity_step)

        avg_w = np.mean(self.wealth[in_group]) if len(in_group) > 0 else 0.0
        cost = 0.01 * avg_w
        self.wealth[in_group] = np.maximum(0.0, self.wealth[in_group] - cost)

        return True

    def _apply_leave(self, group):
        worker_idx = self._worker_indices()
        in_group = worker_idx[self.group_id[worker_idx] == group]
        if len(in_group) == 0:
            return False

        leave_fraction = 0.2
        num_leave = max(1, int(leave_fraction * len(in_group)))
        leave_agents = np.random.choice(in_group, size=num_leave, replace=False)
        self.group_id[leave_agents] = -1

        self.group_unity[group] = max(0.0, self.group_unity[group] - 0.05)
        return True

    def _apply_revolt(self, group, stats_before):
        if self.regime != "capitalist":
            return False, False

        worker_idx = self._worker_indices()
        in_group = worker_idx[self.group_id[worker_idx] == group]
        if len(in_group) == 0:
            return False, False

        p = self._revolution_success_probability(group, stats_before)
        r = np.random.rand()
        if r < p:
            self._revolution_success()
            return True, False
        else:
            self._revolution_fail(group, in_group)
            return False, True

    def _revolution_success_probability(self, group, stats):
        u_g = float(self.group_unity[group])
        g = stats["gini"]
        cr = stats["class_ratio_norm"]
        stress = 0.5 * (g + cr)
        z = self.success_alpha * (u_g - 0.5) + self.success_beta * (stress - 0.5)
        p = 1.0 / (1.0 + math.exp(-z))
        p = max(0.01, min(0.99, p))
        return p

    def _revolution_success(self):
        self.regime = "post_capitalist"
        self.revolution_occurred = True

        self.is_capitalist[:] = False
        self.power_bias = 0.5

        mean_w = np.mean(self.wealth)
        self.wealth = 0.5 * self.wealth + 0.5 * mean_w

    def _revolution_fail(self, group, in_group):
        self.group_unity[group] *= self.unity_decay_failed

        avg_w = np.mean(self.wealth[in_group])
        loss = 0.1 * avg_w
        self.wealth[in_group] = np.maximum(0.0, self.wealth[in_group] - loss)

        self.power_bias = min(0.9, self.power_bias + 0.05)

    def _biased_exchanges(self):
        n = self.num_agents
        for _ in range(self.exchanges_per_step):
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            if i == j:
                continue
            wi, wj = self.wealth[i], self.wealth[j]
            if wi <= 0 and wj <= 0:
                continue

            amt = 0.05 * min(wi, wj)

            if self.is_capitalist[i] and not self.is_capitalist[j]:
                p = self.power_bias
                if np.random.rand() < p:
                    giver, receiver = j, i
                else:
                    giver, receiver = i, j
            elif self.is_capitalist[j] and not self.is_capitalist[i]:
                p = self.power_bias
                if np.random.rand() < p:
                    giver, receiver = i, j
                else:
                    giver, receiver = j, i
            else:
                if np.random.rand() < 0.5:
                    giver, receiver = i, j
                else:
                    giver, receiver = j, i

            transfer = min(amt, self.wealth[giver])
            self.wealth[giver] -= transfer
            self.wealth[receiver] += transfer

    def _equalizing_dynamics(self):
        mean_w = np.mean(self.wealth)
        alpha = 0.2
        self.wealth += alpha * (mean_w - self.wealth)

    def _compute_stats(self):
        w = self.wealth
        g = gini(w)

        worker_idx = self._worker_indices()
        cap_idx = self._capitalist_indices()

        if len(worker_idx) > 0:
            mean_worker = float(np.mean(w[worker_idx]))
        else:
            mean_worker = 0.0

        if len(cap_idx) > 0:
            mean_cap = float(np.mean(w[cap_idx]))
        else:
            mean_cap = mean_worker

        if mean_worker > 0:
            class_ratio = mean_cap / mean_worker
        else:
            class_ratio = 1.0

        class_ratio_norm = class_ratio / (class_ratio + 1.0)

        sorted_w = np.sort(w)
        k = max(1, int(0.1 * len(w)))
        top10 = float(np.sum(sorted_w[-k:]) / np.sum(sorted_w))

        avg_unity = float(np.mean(self.group_unity))
        max_unity = float(np.max(self.group_unity))

        return {
            "gini": g,
            "top10": top10,
            "mean_worker": mean_worker,
            "mean_cap": mean_cap,
            "class_ratio": class_ratio,
            "class_ratio_norm": class_ratio_norm,
            "avg_unity": avg_unity,
            "max_unity": max_unity,
        }

    def _state_from_stats(self, stats):
        return np.array([
            stats["gini"],
            stats["top10"],
            stats["mean_worker"],
            stats["mean_cap"],
            stats["avg_unity"],
            stats["max_unity"]
        ], dtype=np.float32)

    def _compute_reward(
        self,
        stats_before,
        stats_after,
        did_unify,
        did_leave,
        revolt_success,
        revolt_fail
    ):
        mu_w_before = stats_before["mean_worker"]
        mu_w_after = stats_after["mean_worker"]
        g_before = stats_before["gini"]
        g_after = stats_after["gini"]

        delta_mu_w = mu_w_after - mu_w_before
        delta_G = g_before - g_after

        reward = 0.0
        reward += self.lambda_w * delta_mu_w
        reward += self.lambda_g * delta_G

        if revolt_success:
            reward += self.lambda_succ
        if revolt_fail:
            reward -= self.lambda_fail
        if did_unify:
            reward -= self.lambda_u
        if did_leave:
            reward -= self.lambda_l

        return float(reward)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=50000,
        target_update_freq=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self.learn_steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return int(q_vals.argmax().item())

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def train_dqn(
    num_episodes=5000,
    eval_every=20,
    seed=123
):
    env = MarxSimEnv(seed=seed)
    state_dim = 6
    action_dim = env.num_groups * 4

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99
    )

    episode_rewards = []
    eval_returns = []
    eval_episodes_idx = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if ep % eval_every == 0:
            avg_ret = evaluate_policy(env, agent, n_episodes=5)
            eval_returns.append(avg_ret)
            eval_episodes_idx.append(ep)
            print(f"Episode {ep} | train return={total_reward:.2f} | eval return={avg_ret:.2f} | epsilon={agent.epsilon:.3f}")
        else:
            print(f"Episode {ep} | train return={total_reward:.2f} | epsilon={agent.epsilon:.3f}")

    return agent, episode_rewards, (eval_episodes_idx, eval_returns)


def evaluate_policy(env, agent, n_episodes=5):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.01

    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(s)
            s2, r, done, info = env.step(a)
            s = s2
            total += r
        returns.append(total)

    agent.epsilon = old_epsilon
    return float(np.mean(returns))


if __name__ == "__main__":
    agent, train_returns, eval_data = train_dqn(
        num_episodes=5000,
        eval_every=20,
        seed=123
    )

import numpy as np

np.save("train_returns.npy", np.array(episode_rewards))
np.save("eval_episodes.npy", np.array(eval_episodes_idx))
np.save("eval_returns.npy", np.array(eval_returns))
