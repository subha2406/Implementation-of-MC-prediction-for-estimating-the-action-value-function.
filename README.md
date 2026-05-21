# Implementation-of-MC-prediction-for-estimating-the-action-value-function.

## Aim

To implement the Monte Carlo (MC) Prediction algorithm for estimating the action-value function \(Q(s,a)\) using sampled episodes and to analyze the learned action values in a Grid World environment.

---

## Objective

- To understand Monte Carlo Prediction for action-value estimation.
- To estimate the action-value function \(Q(s,a)\).
- To learn state-action values from complete episodes.
- To evaluate the quality of actions under a given policy.

---

## Theory

Monte Carlo Prediction is a model-free reinforcement learning technique used to estimate value functions directly from experience.

The action-value function is defined as:

\[
Q(s,a) = E[G_t \mid S_t=s, A_t=a]
\]

Where:

- \(Q(s,a)\) = Expected return for taking action \(a\) in state \(s\)
- \(G_t\) = Discounted return after time step \(t\)

Monte Carlo methods estimate action values by averaging returns obtained after visiting each state-action pair over many episodes.

---

## Algorithm

### Monte Carlo Prediction for Action-Value Function

1. Initialize:
   - Action-value function \(Q(s,a)\)
   - Returns list for every state-action pair

2. Generate an episode using a policy.

3. For every state-action pair in the episode:
   - Calculate the return \(G\)
   - Store the return for that pair
   - Update \(Q(s,a)\) using the average return

4. Repeat the process for many episodes.

5. Display the estimated action-value function.

---

## Program

```
import numpy as np
from collections import defaultdict
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.9
episodes = 5000

Q = defaultdict(float)

returns = defaultdict(list)

def policy(state):
    return env.action_space.sample()

def generate_episode():

    episode = []

    state, _ = env.reset()

    done = False

    while not done:

        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        episode.append((state, action, reward))

        state = next_state

    return episode

for ep in range(episodes):

    episode = generate_episode()

    G = 0

    visited_pairs = set()

    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = gamma * G + reward

        # First-visit MC
        if (state, action) not in visited_pairs:

            returns[(state, action)].append(G)

            Q[(state, action)] = np.mean(returns[(state, action)])

            visited_pairs.add((state, action))

print("\nAction Value Function:\n")

for key in Q:
    print(f"State-Action {key}: {Q[key]:.3f}")

```

---

## Output


<img width="409" height="708" alt="image" src="https://github.com/user-attachments/assets/9214cad7-3275-4021-898d-0a3aad9eb2bb" />

<img width="315" height="312" alt="image" src="https://github.com/user-attachments/assets/c31d0f9a-14cf-492c-81b7-768b64a2b310" />



---

## Result

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the action-value function \(Q(s,a)\). The expected returns for different state-action pairs were calculated using sampled episodes generated from the environment.

---


---



---

---
