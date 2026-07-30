# Research Extension Roadmap: Hamiltonian Hyperparameter Dynamics (HHD)

> This document maps **10 concrete directions** for extending the HHD framework, organized by immediacy (near-term, medium-term, long-term). Each direction includes motivation, technical approach, key challenges, and relevant recent papers (2024-2025).

---

## How to Read This Document

Your current HHD framework establishes two core ideas:
1. **Hyperparameters as dynamical variables** in an augmented phase space $(\theta, \lambda, p_\theta, p_\lambda)$
2. **Symplectic leapfrog integration + Metropolis acceptance** to co-evolve weights and HPs

Every extension below asks: *"What new class of hyperparameters or training settings can be promoted to dynamical variables, and what domain benefits from structure-preserving HP evolution?"*

---

## Near-Term Extensions (Direct Applicability)

### 1. Physics-Informed Neural Networks (PINNs) — Loss Weight Balancing

> **Status**: Already prototyped in `pinn_overlap/`. Ready for full paper-quality experiments.

**Motivation:** PINNs minimize L = w_r * L_r + w_b * L_b + w_i * L_i, where the loss weights w_r, w_b, w_i must be carefully balanced. Current solutions (SoftAdapt, ReLoBRaLo, NTK weighting) are heuristics with no dynamical-systems grounding. HHD provides momentum, energy conservation, and accept/reject — exactly what these weights need.

**Technical Approach:**
- Promote (log w_r, log w_b, log w_i, log lr) to phase-space variables
- Finite-diff gradients dL/d(log w) (already implemented)
- Benchmark against SoftAdapt, ReLoBRaLo, and fixed weights on canonical PDEs (Heat, Burgers, Navier-Stokes)

**Key Challenge:** The PDE residual loss involves second-order autograd derivatives, making the loss landscape highly non-convex and the HP gradients noisy. Higher mass m_lambda may be needed.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Rathore et al., "Challenges in Training PINNs: A Loss Landscape Perspective," *ICML 2024* | 2024 | Formal analysis of PINN loss landscape pathologies — motivates why structure-preserving HP evolution helps |
| Kiyani et al., "Optimizing the Optimizer for PINNs and KANs," *arXiv:2501.16371* | 2025 | Meta-optimization of PINN optimizers — direct competitor and comparison target |
| Wang et al., "An Expert's Guide to Training Physics-Informed Neural Networks," *arXiv:2308.08468* | 2023 | Comprehensive PINN training guide — establishes the practical HP tuning challenge |
| Kaltsas, "Constrained Hamiltonian Systems and PINNs: Hamilton-Dirac Neural Networks," *Phys. Rev. E* 111 | 2025 | Hamiltonian constraints in PINNs — complementary formulation from the physics side |
| McClenny & Braga-Neto, "Self-Adaptive PINNs," *J. Comput. Phys.* | 2023 | Self-adaptive loss weighting — the baseline your HHD-PINN needs to beat |

---

### 2. Kolmogorov-Arnold Networks (KANs) — Spline Knot & Basis Function Tuning

**Motivation:** KANs replace fixed activation functions with learnable univariate functions (B-splines or wavelets) on network edges. Their hyperparameters — spline order, number of knots, grid resolution, and basis function type — are continuous and directly control expressiveness. Physics-Informed KANs (PIKANs) face the same loss-balancing problem as PINNs but with additional architectural HPs.

**Technical Approach:**
- Promote spline grid resolution and penalty coefficients to phase-space variables
- HHD co-evolves theta_KAN (spline coefficients) and lambda_arch (knot density, regularization strength)
- The Adam to HMC handoff maps naturally to KAN training (Adam warmup then symplectic refinement)

**Key Challenge:** KAN parameter spaces are heterogeneous — some HPs are truly discrete (spline order). Continuous relaxation + rounding (as you already do for n_layers) handles this.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Liu et al., "KAN: Kolmogorov-Arnold Networks," *ICML 2024* | 2024 | Original KAN paper — defines the architecture and HP space |
| Shukla et al., "A Comprehensive Study on PIKANs," *JMLR* | 2025 | Physics-Informed KANs — the intersection of KANs and PDE solving |
| Howard et al., "ChebPIKAN: Chebyshev Polynomial KANs," *AIP Advances* | 2025 | Chebyshev basis KANs for fluid mechanics — domain-specific KAN tuning |
| Kiyani et al., "Optimizing the Optimizer for PINNs and KANs," *arXiv:2501.16371* | 2025 | Joint optimizer optimization for both PINNs and KANs |

---

### 3. Riemannian Manifold Generalization — HHD on Curved HP Spaces

**Motivation:** Your current HHD operates in flat (Euclidean) phase space. But many HP spaces have natural Riemannian geometry — the space of positive-definite matrices (for per-parameter learning rates), the Stiefel manifold (for orthogonal weight constraints), or the simplex (for loss weight proportions that sum to 1). Riemannian HMC (RMHMC) generalizes symplectic integration to curved manifolds.

**Technical Approach:**
- Replace flat leapfrog with Riemannian leapfrog using position-dependent metric tensor G(lambda)
- The Fisher Information Matrix provides a natural metric for HP space
- Automatic step-size adaptation in high-curvature regions (from relativity theory, see Xu & Ge, ICML 2024)

**Key Challenge:** Computing the metric tensor and its Christoffel symbols is expensive. For small HP spaces (k <= 10), the overhead is manageable.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Xu & Ge, "Practical HMC on Riemannian Manifolds via Relativity Theory," *ICML 2024* | 2024 | Relativity-based velocity bounds for stable Riemannian leapfrog |
| Girolami & Calderhead, "Riemann Manifold Langevin and HMC Methods," *JRSS-B* | 2011 | Foundational RMHMC paper — theoretical basis |
| Duruisseaux & Leok, "Variational Formulation on Riemannian Manifolds," *SIAM J. Math. Data Sci.* | 2022 | Already in your bibliography — extends Bregman Hamiltonian work to manifolds |

---

## Medium-Term Extensions (Requires New Machinery)

### 4. Multi-Fidelity HHD — Successive Halving with Hamiltonian Dynamics

**Motivation:** Your current HHD runs all training at full fidelity (full dataset, full epochs). Multi-fidelity methods (Hyperband, BOHB) evaluate cheap, low-fidelity proxies first (fewer epochs, data subsampling) and promote promising configurations. HHD's Metropolis accept/reject is a natural early-stopping criterion — configurations that fail acceptance tests at low fidelity can be pruned immediately.

**Technical Approach:**
- Run HMC trajectories at increasing fidelity levels (fidelity = fraction of training data or epochs)
- Use Hamiltonian energy error |delta H| as a cheap quality signal — high |delta H| at low fidelity predicts poor performance at high fidelity
- Combine with Successive Halving: allocate N parallel HMC chains, halve at each round based on |delta H| ranking

**Key Challenge:** Fidelity changes alter the loss landscape, invalidating the current Hamiltonian. Need to re-randomize momenta or apply a correction term when fidelity changes.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Li et al., "Hyperband: A Novel Bandit-Based Approach to HPO," *JMLR* | 2018 | Foundational multi-fidelity HPO — already in your bibliography |
| Falkner et al., "BOHB: Robust and Efficient HPO at Scale," *ICML 2018* | 2018 | Combines BO with Hyperband — strong baseline |
| Stoll et al., "Automatic Termination for Multi-Fidelity HPO," *OpenReview/NeurIPS* | 2025 | Adaptive stopping criteria — complementary to HHD's accept/reject |
| Mallik et al., "FastBO: Multi-Fidelity BO with Adaptive Fidelity," *NeurIPS* | 2024 | Bayesian-guided fidelity selection — can be combined with HHD |

---

### 5. Bayesian Uncertainty Quantification — HHD as a Posterior Sampler

**Motivation:** When your HHD temperature T is set to a finite value (not 1e9), the Metropolis acceptance criterion samples from the posterior p(theta, lambda | D) proportional to exp(-H/T). This means HHD is already a Bayesian sampler in disguise — you just haven't used it that way. By collecting accepted (theta, lambda) samples rather than discarding them, you get a posterior distribution over *both* weights and hyperparameters for free.

**Technical Approach:**
- Set T to a physically meaningful value (e.g., T = 1) instead of 1e9
- Collect M accepted samples {(theta_i, lambda_i)} from the HMC chain
- Predictive uncertainty: Var[y] = (1/M) sum Var[f_theta_i(x)] (model uncertainty from theta diversity) + HP uncertainty from lambda diversity
- This provides **joint weight-HP uncertainty** — something no existing BNN method offers

**Key Challenge:** Finite-temperature HMC in high dimensions requires longer chains and careful tuning of T, epsilon, and L. Your existing persistent-momentum and step-size controller help.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Arbel et al., "Scalable Bayesian Monte Carlo," *NeurIPS 2024* | 2024 | Scalable BNN inference — comparison target for HHD-as-sampler |
| Cobb & Jalaian, "Scaling HMC for Bayesian Neural Networks," *Comput. Sci. Rev.* | 2025 | Practical split-HMC for large BNNs — directly relevant techniques |
| Neal, "MCMC Using Hamiltonian Dynamics," *Handbook of MCMC* | 2011 | Already in your bibliography — the theoretical foundation |
| Wilson & Izmailov, "Bayesian Deep Learning and a Probabilistic Perspective," *NeurIPS 2020* | 2020 | Motivates why joint weight-HP uncertainty matters |

---

### 6. Diffusion Models — Noise Schedule & Guidance Scale Tuning

**Motivation:** Diffusion models (DDPM, score-based) have continuous, high-sensitivity hyperparameters: noise schedule beta(t), guidance scale w_cfg, number of denoising steps T, and learning rate schedule. The noise schedule is a function that controls the signal-to-noise ratio at each timestep — its shape dramatically affects sample quality (FID/IS scores). Current practice uses fixed linear/cosine schedules or expensive grid search.

**Technical Approach:**
- Parameterize the noise schedule as a low-dimensional vector (e.g., 10 control points of a spline) and promote to phase-space variables
- HHD co-evolves the diffusion model weights and the noise schedule spline coefficients
- The denoising loss L_DSM = E[||s_theta(x_t, t) - grad log p(x_t|x_0)||^2] plays the role of the potential V

**Key Challenge:** The noise schedule affects every training step, so its gradient signal is aggregated over the entire diffusion trajectory — potentially noisy.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Kingma & Gao, "Understanding Diffusion Objectives as SNR-Weighted Score Matching," *NeurIPS 2024* | 2024 | Log-SNR perspective on noise schedules — provides the optimization target |
| Chen et al., "Importance of Noise Scheduling for Diffusion Models," *CVPR 2024* | 2024 | Empirical study of schedule sensitivity — motivates automated tuning |
| Chen, "On the Optimal Control of Noise Schedules," *arXiv* | 2025 | Optimal control framing — complementary to HHD's dynamical-systems view |

---

### 7. Reinforcement Learning — Reward Shaping & Policy HPs

**Motivation:** RL training involves multiple interacting hyperparameters: reward scale, discount factor gamma, entropy coefficient alpha, clipping ratio epsilon_clip (in PPO), and GAE parameter lambda_GAE. These are continuous, interact strongly with each other, and their optimal values change during training (e.g., entropy should decrease as the policy matures). HHD's momentum-based evolution naturally captures such time-varying optima.

**Technical Approach:**
- Promote (log alpha, gamma, epsilon_clip, lambda_GAE) to HHD phase-space variables
- The RL loss (e.g., PPO's clipped surrogate objective) serves as the potential V
- HMC accept/reject prevents catastrophic HP changes that destabilize policy learning

**Key Challenge:** RL losses are extremely noisy (high-variance policy gradient estimates). May need to average loss over multiple rollouts before computing HP gradients.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Eimer et al., "Hyperparameters in RL and How To Tune Them," *ICML 2023* | 2023 | Comprehensive RL HPO study — defines the problem space |
| Parker-Holder et al., "Automated RL: A Survey and Open Problems," *JAIR* | 2022 | AutoRL survey — establishes why automated RL HP tuning is critical |
| Moerland et al., "Model-based RL: A Survey," *Found. & Trends in ML* | 2023 | Covers Hamiltonian dynamics for model-based RL control |

---

## Long-Term Extensions (New Research Programs)

### 8. Federated & Distributed HHD — Privacy-Preserving HP Co-Evolution

**Motivation:** In federated learning, each client trains locally with potentially different optimal HPs (due to non-IID data). HHD can be distributed: each client runs its own HMC chain on (theta_local, lambda_local), and a server aggregates HP proposals across clients. The Hamiltonian framework naturally supports this because each client's energy is additive — the total Hamiltonian is H = sum_c H_c.

**Technical Approach:**
- Federated HHD: clients share HP proposals (not data) -> server accepts/rejects based on aggregate delta H
- Privacy: only lambda values and scalar delta H are communicated — no gradient or data leakage
- Non-IID robustness: per-client mass m_lambda^(c) controls how strongly each client's HPs resist server-proposed changes

**Key Challenge:** Communication efficiency — HMC proposals require synchronization. Asynchronous HMC variants or periodic synchronization (every K rounds) can reduce overhead.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Deng et al., "FA-HMC: Federated Averaging Stochastic HMC," *arXiv* | 2024 | Federated HMC for Bayesian FL — directly relevant algorithm |
| Khodak et al., "Federated Hyperparameter Tuning: Challenges, Baselines, and Connections," *NeurIPS 2021* | 2021 | Formal framework for federated HPO — establishes the problem |
| Guo et al., "PRIVTUNA: Privacy-Preserving HP Tuning," *IEEE TIFS* | 2024 | Homomorphic encryption for federated HPO — complementary privacy approach |

---

### 9. Graph Neural Networks — Message-Passing Architecture Search

**Motivation:** GNNs have a unique HP space: number of message-passing layers, aggregation function, attention heads, neighborhood sampling strategy, and skip-connection patterns. Over-smoothing (features becoming indistinguishable after too many layers) makes depth a critical continuous HP. HHD can navigate this space with its reflection boundary conditions preventing the model from "smoothing to death."

**Technical Approach:**
- Promote GNN-specific HPs (depth as continuous relaxation, dropout per layer, attention temperature) to phase space
- The message-passing loss + downstream task loss serves as potential V
- Reflection BCs on depth in [1, K_max] naturally prevent over-smoothing

**Key Challenge:** GNN training involves irregular-shaped batches (subgraph sampling), making loss evaluation noisy.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| GNN-Diff, "Latent Diffusion for GNN Configuration," *arXiv* | 2024 | Generative approach to GNN tuning — comparison target |
| Choi et al., "GNN-NAS: A Comprehensive Survey," *Res. Gate* | 2024 | Survey of GNN architecture search — positions HHD in context |
| You et al., "Design Space for GNNs," *NeurIPS 2020* | 2020 | Formal GNN design space — defines what HPs to co-evolve |

---

### 10. Meta-Learning Warm-Starting — Transfer HHD Across Tasks

**Motivation:** Currently, HHD starts from scratch for each new problem. But the dynamics of good HP evolution are transferable — if HHD learns that "boundary condition weight should increase in the first 50 epochs" for one PDE, that knowledge should transfer to similar PDEs. Meta-learning can learn an optimal initial HP state (lambda_0, p_lambda_0) and mass schedule m_lambda(t) from a distribution of tasks.

**Technical Approach:**
- Outer loop: meta-learn (lambda_0, p_lambda_0, m_lambda) over a task distribution (e.g., family of PDEs with varying coefficients)
- Inner loop: standard HHD training on each task, starting from meta-learned initialization
- This is MAML applied to the HP dynamics rather than the model weights

**Key Challenge:** Requires a distribution of related tasks, not just a single problem. PINN families (PDEs with varying coefficients) are ideal.

**Relevant Papers:**

| Paper | Year | Key Contribution |
|:------|:-----|:-----------------|
| Yang et al., "muTransfer: Width-Based HP Transfer," *ICML 2022* | 2022 | HP transfer across model scales — analogous to HHD transfer across tasks |
| Antoniou & Storkey, "Learning to Learn via Meta-Learning," *ICML* | 2019 | Adaptive inner-loop HPs — the meta-learning foundation |
| MetaPEFT, "Meta-Learning for Parameter-Efficient Fine-Tuning," *CVPR 2025* | 2025 | Meta-learned per-module learning rates — directly analogous to meta-learned HHD mass |

---

## Summary Matrix

| # | Direction | HP Variables | Domain | Effort | Impact |
|:--|:----------|:-------------|:-------|:-------|:-------|
| 1 | PINNs loss balancing | w_r, w_b, w_i, lr | Scientific ML | Low | High |
| 2 | KAN tuning | Spline order, grid density | Scientific ML | Low | Medium |
| 3 | Riemannian HHD | Any HP on curved manifold | Theory | Medium | High |
| 4 | Multi-fidelity HHD | All + fidelity level | AutoML | Medium | High |
| 5 | Bayesian UQ | theta + lambda posterior | Safety-critical ML | Medium | High |
| 6 | Diffusion models | Noise schedule, guidance | Generative AI | Medium | Medium |
| 7 | RL reward/policy HPs | alpha, gamma, epsilon_clip | RL/Control | Medium | Medium |
| 8 | Federated HHD | Per-client HPs | Distributed ML | High | High |
| 9 | GNN arch search | Depth, aggregation, attention | Graph ML | Medium | Medium |
| 10 | Meta-learning warm-start | lambda_0, p_lambda_0, m_lambda | Transfer learning | High | High |

---

## Recommended Reading Order

If you want to go deep on the most impactful directions:

1. **Start with Direction 1 (PINNs)** — you already have the code. Read Rathore et al. (ICML 2024) and Wang et al. (2023) for the loss landscape analysis.
2. **Then Direction 5 (Bayesian UQ)** — it requires the least new code (just change T from 1e9 to 1.0 and collect samples). Read Cobb & Jalaian (2025).
3. **Then Direction 4 (Multi-Fidelity)** — combine HHD with Hyperband. Read Li et al. (2018) which you already cite.
4. **Then Direction 3 (Riemannian)** — the most theoretically rich extension. Read Xu & Ge (ICML 2024) for the practical algorithm.

**The single most publishable extension** is Direction 5 (Bayesian UQ): the observation that HHD at finite temperature provides a joint weight-HP posterior is novel and directly follows from your existing framework. No other HPO method provides uncertainty over hyperparameters, only over weights.
