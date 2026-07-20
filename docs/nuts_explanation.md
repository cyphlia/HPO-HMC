# No-U-Turn Sampler (NUTS) for Joint Hamiltonian Hyperparameter Dynamics (HHD)

This document provides a comprehensive explanation of the **No-U-Turn Sampler (NUTS)** integration in the HHD framework, as implemented in [nuts_benchmark.py](file:///c:/Minor Project/Model/current/nuts_benchmark.py). 

---

## 1. Motivation and Problem Context

In standard **Hamiltonian Monte Carlo (HMC)**, exploration is simulated using a physical analogy where parameters act as positions in a potential well, and are updated using a symplectic (leapfrog) integrator. This requires configuring two hyper-hyperparameters:
*   **Step size ($\epsilon$):** Determines the discretization granularity.
*   **Number of leapfrog steps ($L$):** Dictates how long the trajectory is simulated before making a Metropolis-Hastings proposal.

This fixed-trajectory approach ($L$ steps) suffers from two fundamental limitations:
1.  **Under-exploration ($L$ too small):** The trajectory is too short to explore the target distribution effectively, leading to random-walk-like behavior.
2.  **U-turning/Over-computation ($L$ too large):** The trajectory is simulated for too long, causing the particle to orbit and double back on itself (returning close to its starting position). This wasting of computational power is especially costly in HHD, where each leapfrog step requires expensive neural network parameter backpropagation and finite-difference hyperparameter gradients.

### The NUTS Solution
Introduced by Hoffman & Gelman (2014), the **No-U-Turn Sampler (NUTS)** automates the selection of trajectory length ($L$). It adaptively grows the trajectory (building a binary tree of leapfrog steps) and stops as soon as it detects that continuing the simulation would bring the state back towards its starting position—i.e., when it begins to make a **U-turn**.

---

## 2. NUTS Mathematical Foundations

### 2.1 Recursive Binary Tree Building
Instead of simulating a fixed number of steps, NUTS simulates trajectories by doubling their length at each step of a recursive tree-building process.
*   At depth $d = 0$, the tree consists of $2^0 = 1$ leapfrog step.
*   At depth $d$, the tree consists of $2^d$ leapfrog steps.
*   The expansion direction—either forward or backward in virtual time—is determined by a random coin flip ($\text{direction} \in \{-1, 1\}$).

```mermaid
graph TD
    A[Start State: Tree Depth 0] --> B{Direction Flip}
    B -- Forward +1 -- > C[Build Right Subtree: 2^d steps]
    B -- Backward -1 --> D[Build Left Subtree: 2^d steps]
    C --> E[Check U-Turn on Full Tree]
    D --> E
    E -- No U-Turn & depth < max_depth --> F[Double Tree Depth: d = d+1]
    F --> B
    E -- U-Turn Detected OR max_depth reached --> G[Terminate Integration & Sample Candidate]
```

### 2.2 The U-Turn Criterion
A U-turn is detected when the dot product between the displacement vector (from the start of the trajectory to the end) and the momentum vectors at the boundaries becomes negative:
$$\langle q^+ - q^-, p^-\rangle < 0 \quad \text{or} \quad \langle q^+ - q^-, p^+\rangle < 0$$
where:
*   $q^-$ and $q^+$ are the leftmost and rightmost position states in the simulated trajectory.
*   $p^-$ and $p^+$ are the leftmost and rightmost momentum states.

When either dot product is negative, it indicates that the momentum vector is pointing back towards the opposite end of the trajectory, signifying a U-turn.

### 2.3 Transition Probability (Slice & Multinomial Sampling)
To maintain detailed balance, NUTS uses a slice variable:
$$u \sim \text{Uniform}(0, \exp(-H(q, p)))$$
A candidate state is chosen from the set of states in the tree that satisfy the slice threshold ($H(q, p) - H(q_0, p_0) < \Delta_{\max}$). In the naive NUTS implementation (Algorithm 3 of the paper), states are sampled uniformly from the valid subtrees, preserving the target distribution without requiring the pre-specification of $L$.

---

## 3. Joint Phase Space Adaptation for HHD

In **Hamiltonian Hyperparameter Dynamics (HHD)**, the target space is a joint space of:
1.  **Network Weights ($\theta$):** High-dimensional, updated via backpropagation. Assigned mass $m_\theta$ and momentum $p_\theta$.
2.  **Hyperparameters ($\lambda$):** Continuous parameters (e.g., learning rate, dropout, weight decay), updated via finite-difference approximations of the loss. Assigned mass $m_\lambda$ and momentum $p_\lambda$.

This creates a joint state space:
$$\mathbf{x} = [\theta, \lambda], \quad \mathbf{p} = [p_\theta, p_\lambda]$$

### The Joint U-Turn Criterion
If the U-turn check is evaluated strictly over the combined vector space, high-frequency oscillations in one component (e.g., network weights) might satisfy the U-turn condition early and stop integration before the slower component (e.g., hyperparameters) has had a chance to explore.

To address this, our implementation in [NUTSIntegrator](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L214) evaluates the U-turn condition **separately for the weight and hyperparameter subsystems**. A U-turn is triggered if *either* subsystem begins to double back:

#### 1. Hyperparameter Subsystem U-Turn Check
$$\text{dot}_{\lambda}^- = \sum_{k} (\lambda_k^+ - \lambda_k^-) \cdot p_{\lambda, k}^-$$
$$\text{dot}_{\lambda}^+ = \sum_{k} (\lambda_k^+ - \lambda_k^-) \cdot p_{\lambda, k}^+$$

#### 2. Weight Subsystem U-Turn Check (Aggregated)
$$\text{dot}_{\theta}^- = \sum_{i} (\theta_i^+ - \theta_i^-) \cdot p_{\theta, i}^-$$
$$\text{dot}_{\theta}^+ = \sum_{i} (\theta_i^+ - \theta_i^-) \cdot p_{\theta, i}^+$$

#### 3. Stopping Decision
$$\text{Stop} = (\text{dot}_{\lambda}^- < 0) \lor (\text{dot}_{\lambda}^+ < 0) \lor (\text{dot}_{\theta}^- < 0) \lor (\text{dot}_{\theta}^+ < 0)$$

This joint criterion ensures that both network weight optimization and hyperparameter search co-evolve constructively without stalling each other.

---

## 4. Codebase Implementation Details

The implementation is split into two primary classes in [nuts_benchmark.py](file:///c:/Minor Project/Model/current/nuts_benchmark.py):

### 4.1 `NUTSIntegrator`
The [NUTSIntegrator](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L214) class manages the adaptive trajectory simulation:

*   **[_leapfrog_step()](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L253):** Computes model gradients $\nabla_\theta \mathcal{L}$ and hyperparameter finite-difference gradients $\nabla_\lambda \mathcal{L}$, performing a symplectic half-step momentum update, full-step position update, and half-step momentum update.
*   **[_check_uturn()](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L304):** Computes the dot products for both the hyperparameter and weight subsystems, excluding frozen hyperparameters (like network layer depth and width which are locked during Phase 2).
*   **[integrate()](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L346):** Implements the main tree-doubling loop. At each iteration:
    1.  Chooses a random direction (forward/backward).
    2.  Restores the corresponding endpoint (`snap_minus` or `snap_plus`).
    3.  Runs $2^{\text{depth}}$ leapfrog steps.
    4.  Performs energy checks against the initial Hamiltonian $H_{\text{init}}$ to identify valid states.
    5.  Performs a multinomial selection to accept/reject the candidate state from the new subtree.
    6.  Checks if a U-turn has occurred across the full tree bounds. If so, it halts and restores the best state found.

### 4.2 `HamiltonianMCMC_NUTS`
The [HamiltonianMCMC_NUTS](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L450) class wraps the integrator in a Metropolis-Hastings framework:

*   **[_refresh_momenta()](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L474):** Implements a partial momentum refresh governed by the parameter $\alpha$ (momentum refresh rate):
    $$p_{\text{new}} = \sqrt{1 - \alpha} p_{\text{old}} + \sqrt{\alpha} \eta, \quad \eta \sim \mathcal{N}(0, m)$$
    This preserves the momentum direction across proposals to maintain optimization directionality while periodically refreshing kinetic energy to avoid stagnation.
*   **[propose()](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L496):**
    1.  Backs up the current weights, hyperparameters, and momenta.
    2.  Refreshes the momenta.
    3.  Calls the [NUTSIntegrator](file:///c:/Minor Project/Model/current/nuts_benchmark.py#L214)'s `integrate` method.
    4.  Computes the Metropolis-Hastings acceptance probability based on the total Hamiltonian energy $H = T_\theta + T_\lambda + V(\theta, \lambda)$.
    5.  On acceptance, commits the proposed weights and hyperparameters. On rejection, rolls back and reverses the momentum direction.

---

## 5. Empirical Benchmarks & Performance Trade-offs

The performance of NUTS is evaluated alongside standard fixed-leapfrog HMC variants on the **Wisconsin Diagnostic Breast Cancer** task. The summarized results from [nuts_comparison_summary.json](file:///c:/Minor Project/Model/current/results/breast_cancer/nuts_comparison_summary.json) are presented below:

| Method | Description | Test AUROC | Test Accuracy (%) | Malignant Recall (%) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HMC-Original** | Fixed $L=4$, single-batch, $T=10^9$ | $0.9974 \pm 0.0018$ | $96.78 \pm 0.83$ | $96.09 \pm 1.12$ | $3.73$ |
| **HMC-Fixed** | Fixed $L=4$, full-batch, $T=100$, partial refresh | $0.9987 \pm 0.0012$ | $\mathbf{98.83 \pm 0.41}$ | $98.45 \pm 1.10$ | $\mathbf{3.19}$ |
| **Method C (NUTS)** | Adaptive trajectory, full-batch, partial refresh | $0.9974 \pm 0.0019$ | $98.25 \pm 1.24$ | $\mathbf{99.22 \pm 1.10}$ | $5.41$ |

### Performance Analysis

1.  **Clinical Quality (Malignant Recall):**
    In clinical diagnostic support, missing a positive case (malignant case) is far costlier than a false alarm. NUTS achieves the **highest malignant recall** ($\mathbf{99.22\%}$), matching the performance of computationally heavy Random Search baselines while executing much faster.
    
2.  **Adaptive Computational Overhead:**
    NUTS runs slightly slower ($5.41\,\text{s}$) than the fixed-step HMC ($3.19\,\text{s}$). This is because NUTS adaptively simulates deeper trajectories (increasing leapfrog steps) when momentum is high and no U-turn is detected. 
    However, this minor wall-clock difference is heavily compensated by the **elimination of manual tuning** for the trajectory step parameter $L$.

3.  **Stability:**
    By checking U-turn criteria separately for weights and hyperparameters, the system remains stable and does not suffer from premature termination.
