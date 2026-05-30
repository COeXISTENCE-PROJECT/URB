# Multi-Arm Bandit Experiments in URB

## 1. Project Background and Objectives

The **CoEXISTENCE** initiative investigates the evolving dynamics of urban traffic as the proportion of Autonomous Vehicles (AVs) rises alongside traditional human-driven cars. While human commuters naturally adapt their daily routes to minimize travel time, AV fleets will increasingly rely on automated, data-driven algorithms to continuously compute the fastest paths based on live traffic conditions.

To achieve this, researchers are heavily exploring the application of Multi-Agent Reinforcement Learning (MARL). However, deploying independent MARL agents in a shared routing environment often leads to macroscopic network instability. Current state-of-the-art MARL models frequently suffer from two critical flaws in this domain: they either get trapped in local optima (yielding chronically suboptimal route distributions), or they do eventually find the optimal solution but suffer from severe sample inefficiency, requiring an impractically long training period.

The core objective of this internship research is to diagnose the root causes of this algorithmic instability and engineer novel routing policies that drastically reduce the convergence time. These newly designed AV routing algorithms are rigorously evaluated using the **Urban Routing Benchmark (URB)** framework (documented at [https://arxiv.org/abs/2505.17734](https://arxiv.org/abs/2505.17734)).

---

## 2. The Shift to Multi-Armed Bandits and Hierarchical Routing

Given the convergence bottlenecks and destabilization observed in traditional deep MARL approaches, this research explores **Multi-Armed Bandit (MAB)** architectures and **Hierarchical Routing** frameworks as robust alternatives.

### The Case for Bandits
In multi-agent urban routing, the environment is highly non-stationary because all vehicles are simultaneously updating their behaviors. Complex Deep RL networks often struggle to stabilize under these constantly shifting reward distributions. 

MAB algorithms (such as Upper Confidence Bound or Bandit-based Policy Gradients) sidestep this by treating the routing problem as a stateless (or highly simplified) exploration-exploitation dilemma. By stripping away complex, sequential temporal state tracking, Bandits provide a much more stable, mathematically bounded approach to finding a Nash equilibrium. This allows the AVs to converge on optimal, congestion-free route distributions significantly faster and with far less computational overhead.

### The Need for Hierarchical Experiments
Furthermore, scaling pure RL to massive, city-wide traffic networks is computationally prohibitive due to the exponentially growing action space. Hierarchical architectures decouple the routing problem to solve this:
1. **Macro-Routing (Global):** A higher-level planner dictates broad corridors or regional waypoints.
2. **Micro-Routing (Local):** Lightweight Bandit agents operate at specific bottlenecks or intersections to make real-time, localized routing decisions.

This separation of concerns drastically shrinks the action space for the RL agents, serving as a highly promising method for accelerating convergence in the URB environment.

---

## 3. Algorithm Overview: Bandit REINFORCE

To test the efficacy of the Bandit approach, we implemented a **Bandit REINFORCE** algorithm. This is an independent policy gradient model utilizing a PyTorch Multi-Layer Perceptron (MLP), replacing deterministic algorithms with a differentiable, stochastic approach.

### Action Selection
The agent’s neural network outputs a vector of scores (logits) for each available route. These logits are scaled by a **Temperature ($\tau$)** parameter to control the sharpness of the distribution, and are then converted into probabilities using a Softmax function:

$$ \pi(a|s) = \frac{\exp(Z_a / \tau)}{\sum_{i} \exp(Z_i / \tau)} $$

### Policy Update (REINFORCE)
Once the vehicle completes its route, it receives a reward $R$ (negative travel time). The weights of the neural network $\theta$ are updated using gradient ascent, scaled by the **Learning Rate ($\alpha$)**:

$$ \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a|s) (R - b) $$

### Exploration via Entropy
To prevent premature convergence to sub-optimal routes, an **Entropy Bonus** is added to the objective function, scaled by the **Entropy Coefficient ($\beta$)**:

$$ H(\pi) = - \sum_{a} \pi(a|s) \log \pi(a|s) $$

---

## 4. Experimental Setup

We conducted a massive hyperparameter sweep of the Bandit REINFORCE algorithm to identify the parameters that yield the fastest and most stable convergence. The sweep consisted of **54 total parallel jobs**, evaluating every combination of the following parameters across **3 random seeds**:

* **Learning Rate ($\alpha$):** `3e-4`, `1e-3`
* **Entropy Coefficient ($\beta$):** `0.01`, `0.05`, `0.10`
* **Temperature ($\tau$):** `0.5`, `1.0`, `2.0`
* **Random Seeds:** `42`, `123`, `7`

Each job executed 500 training episodes and 100 testing episodes on the `ingolstadt_custom` network map. Prior to execution, a prerequisite TraCI bug was patched (ensuring `env.simulator.sumo_connection` was used) to guarantee the network received live, real-time traffic data rather than neutral constants.

---

## 5. Results and Rankings

The results demonstrate the extreme stability of the Bandit REINFORCE algorithm in this topology. The average travel time for Autonomous Vehicles (CAVs) across all configurations fell into a tightly clustered band between **4.328 seconds** and **4.403 seconds**.

### 🏆 Top 5 Configurations
Ranked by the lowest average CAV travel time across all 3 seeds.

| Rank | Learning Rate | Entropy | Temperature | Average CAV Travel Time (s) |
|------|---------------|---------|-------------|-----------------------------|
| **1**| **1e-3**      | **0.05**| **0.5**     | **4.328**                   |
| 2    | 1e-3          | 0.10    | 1.0         | 4.363                       |
| 3    | 1e-3          | 0.10    | 0.5         | 4.365                       |
| 4    | 3e-4          | 0.01    | 0.5         | 4.373                       |
| 5    | 1e-3          | 0.01    | 1.0         | 4.374                       |

### 📉 Bottom 5 Configurations

| Rank | Learning Rate | Entropy | Temperature | Average CAV Travel Time (s) |
|------|---------------|---------|-------------|-----------------------------|
| 50   | 3e-4          | 0.10    | 1.0         | 4.393                       |
| 51   | 3e-4          | 0.01    | 2.0         | 4.395                       |
| 52   | 1e-3          | 0.01    | 2.0         | 4.397                       |
| 53   | 3e-4          | 0.05    | 0.5         | 4.401                       |
| 54   | 3e-4          | 0.01    | 1.0         | 4.403                       |

---

## 6. Visual Analysis

### The Best Configuration
**Parameters:** `lr=1e-3`, `entropy=0.05`, `temperature=0.5`
Notice the tight, highly stable convergence band during the testing phase.

**Travel Times:**
![Best Travel Times](results/sweep_bandit/lr_1e-3_ent_0.05_temp_0.5_seed_42/plots/travel_times.png)

**Rewards:**
![Best Rewards](results/sweep_bandit/lr_1e-3_ent_0.05_temp_0.5_seed_42/plots/rewards.png)

---

### The Worst Configuration
**Parameters:** `lr=3e-4`, `entropy=0.01`, `temperature=1.0`
While the convergence still succeeds, it is noticeably noisier with wider variance during both training and testing.

**Travel Times:**
![Worst Travel Times](results/sweep_bandit/lr_3e-4_ent_0.01_temp_1.0_seed_42/plots/travel_times.png)

**Rewards:**
![Worst Rewards](results/sweep_bandit/lr_3e-4_ent_0.01_temp_1.0_seed_42/plots/rewards.png)
