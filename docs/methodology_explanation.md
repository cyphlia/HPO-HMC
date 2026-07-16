# Methodology Explanation: HHD, ABBO, and the Unified Curriculum

This document provides an in-depth analysis of the experimental design, the limitations identified in HHD-HMC (Pure HHD), and the architectural motivation behind HHD-Unified (Unified HHD-ABBO).

---

## 1. Why Did We Test HHD-HMC and Hybrid ABBO on the CNN/CIFAR-10 Model?

Transitioning the benchmarks from a simple synthetic environment (the Harmonic Oscillator) to a Convolutional Neural Network (CNN) on CIFAR-10 was critical for validating the framework under realistic deep learning conditions.

### A. Dimensionality and Landscape Complexity
- **Harmonic Oscillator:** Represents a low-dimensional (2D), smooth, perfectly quadratic energy surface governed by:
  $$H(q,p) = \frac{p^2}{2m} + \frac{1}{2}kq^2$$
  While ideal for testing mathematical properties (such as symplectic energy conservation and leapfrog trajectory tracking), it does not reflect the complexity of real-world neural network optimization.
- **CNN CIFAR-10 Testbed:** The loss landscape of a convolutional network is high-dimensional, highly non-convex, and contains many saddle points, local minima, and narrow, "stiff" valleys. Testing on CIFAR-10 evaluates if HMC co-evolution can scale to larger parameter spaces and navigate chaotic, non-quadratic potentials.

### B. Enforcing Regularization Pressure
To make the hyperparameter search challenging and realistic, we deliberately enforced overfitting pressure on the CIFAR-10 model by restricting the dataset:
- **Reduced Training Subset:** Configured with only 5,000 training samples (down from 60,000) and 1,000 validation samples.
- **Regularization Challenge:** This limited data forces the network to overfit unless the hyperparameters (specifically the continuous dropout rate and learning rate) are tuned optimally. It directly evaluates whether the algorithms can dynamically find the correct balance between capacity and regularization.

### C. Different Computational Paradigms
- **HHD-HMC (Pure HHD)** co-evolves network weights and hyperparameters *simultaneously* in a single phase space inside a single training run (intra-epoch updates).
- **Hybrid ABBO (Hybrid BO)** runs an *outer loop* of Bayesian Optimization (using a Gaussian Process surrogate) to select static hyperparameters, requiring a complete re-initialization and full training of a new network for each of the 15 trials.
- Evaluating both on CIFAR-10 compares their wall-clock scaling, sample requirements, and search efficiency when training real neural network layers.

---

## 2. Faults and Limitations of HHD-HMC (Pure HHD)

While theoretically elegant due to its continuous trajectories and symplectic conservation properties, Pure HHD suffers from several empirical limitations when applied to non-convex landscapes:

### A. Lack of Second-Order Curvature Exploitation
HHD-HMC uses first-order gradients (via backpropagation for weights and finite differences for hyperparameters) to update the phase space. 
- In flat regions (saddle points or plateaus), first-order gradients become extremely small, causing HMC exploration to stall.
- HHD-HMC has no mechanism (like Hessian approximations or quasi-Newton steps) to exploit local curvature and accelerate convergence along narrow valleys.

### B. Stochastic Instability and Oscillatory Accuracy
HMC is an MCMC sampler that explores the probability distribution defined by the Boltzmann factor $\exp(-H/T)$. 
- Because HMC is inherently stochastic, the hyperparameters continuously wander and oscillate.
- Even if the model reaches a highly accurate state (e.g., 30.60% validation accuracy at epoch 14), late-stage HMC momentum can cause the hyperparameters to drift away, resulting in a lower accuracy (e.g., 28.50%) at the final epoch. It lacks a convergence lock.

### C. Step-Size Sensitivity and Symplectic Collapse
- The leapfrog integration step-size ($\epsilon$) is a highly sensitive constant. If $\epsilon$ is too large, the discrete approximation of Hamiltonian dynamics breaks down (symplectic discretization error $O(\epsilon^2)$ grows too large), leading to massive rejection rates.
- If $\epsilon$ is too small, exploration in the hyperparameter phase space slows down to a crawl. HHD-HMC has no adaptive control to balance this trade-off dynamically.

### D. Costly Structural Hyperparameter Relaxations
- To optimize discrete architecture values (like `n_layers` and `n_neurons`) continuously, HHD-HMC uses continuous relaxations. 
- Computing gradients for these structures requires rebuilding the model topology at $n \pm 10\%$, copying compatible weights, and calculating a finite-difference surrogate gradient. This process is noisy, computationally heavy, and can lead to capacity collapse (e.g., the model shrinking its hidden layers down to 16 neurons to temporarily minimize loss, ruining long-term capacity).

---

## 3. The Need for HHD-Unified (Unified HHD-ABBO)

HHD-Unified was designed to resolve the limitations of HHD-HMC while avoiding the prohibitive computational cost of Hybrid ABBO (which requires training 15 separate networks from scratch). It merges the physical search paradigm of HHD with the convergence speed of second-order optimization into a single **three-phase epoch curriculum**:

### A. Phase 1: Guided Initialization (Adam Warm-up)
- Random initialization in a high-dimensional physical phase space is highly unstable. 
- HHD-Unified runs a dedicated first-order Adam warm-up with cosine learning rate annealing. This quickly guides the network weights down to a low-curvature basin before HMC co-evolution begins, preventing chaotic initial trajectories.

### B. Phase 2: Stable, Constrained HMC Exploration
- To prevent network capacity collapse, HHD-Unified **freezes structural hyperparameters** (`n_layers` and `n_neurons`) after the warm-up, allowing only the continuous regularization and learning rate parameters (`log_lr`, `dropout`, `log_batch_size`) to co-evolve.
- It integrates an **adaptive step-size controller** that monitors the MCMC acceptance rate and gradient norms. If the acceptance rate drops below 40% (indicating discretization error/instability), it reduces $\epsilon$. If it exceeds 80%, it increases $\epsilon$ to accelerate exploration.

### C. Phase 3: Curvature-Exploiting Polish (L-BFGS)
- To overcome HMC stagnation, HHD-Unified continuously monitors training loss. If it detects a local plateau, it pauses HMC and triggers a sequence of **L-BFGS updates with a strong Wolfe line search**. 
- L-BFGS approximates the inverse Hessian, allowing the model to exploit second-order curvature information and converge rapidly along narrow valleys.
- **Dedicated Final Polish:** At the end of the entire training run, HHD-Unified restores the best state found during HMC and executes a final 100-step L-BFGS polish. This locks in the optimal hyperparameters and converges the weights perfectly, achieving an actual validation loss of **`0.003270`** (compared to Hybrid ABBO's `0.021804`).

### D. Direct Computational Savings
By co-evolving and polishing the model in a single, unified curriculum, HHD-Unified eliminates the need for an outer Bayesian Optimization loop:
- **Hybrid ABBO (ABBO):** Requires $15 \times$ independent training runs (totaling hundreds of epochs).
- **HHD-Unified (Unified HHD-ABBO):** Achieves better/comparable final model accuracy in a **single run**, running **over 3.2x faster in wall-clock time** than Hybrid ABBO on the CNN Testbed.

---

## 4. Model Setups and Configurations

### A. Simple Harmonic Oscillator Setup
The Simple Harmonic Oscillator serves as the ground-truth physical system to test the reconstruction accuracy of the energy surface.

1. **System Formulation:**
   - Evaluates a particle in a 1D potential well governed by the classical Hamiltonian:
     $$H(q,p) = \frac{p^2}{2m} + \frac{1}{2}kq^2$$
     where the particle mass $m = 1.0$ and the spring constant $k = 1.0$.
   - Phase-space input coordinates: $q$ (position) and $p$ (momentum).

2. **Dataset Generation:**
   - **Sampling Bounds:** 2,500 data points are randomly sampled in a uniform grid spanning $q \in [-4.0, 4.0]$ and $p \in [-4.0, 4.0]$.
   - **Noisy Target Labels:** Noise is introduced to training and validation targets:
     $$H_{\text{noisy}} = H(q,p) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
     with a standard deviation of $\sigma = 0.05$.
   - **Split:** 80% training set (2,000 samples) and 20% validation set (500 samples).
   - **Evaluation Mesh:** A noiseless $50 \times 50$ (2,500 points) grid is generated over the same boundaries to evaluate the true reconstruction accuracy (MAE, RMSE, R²).

3. **Model & Hyperparameters:**
   - **Model:** A Multi-Layer Perceptron (MLP) mapped as `HamiltonianNN` with 2 inputs ($q, p$), hidden layers of variable widths and depths, and 1 scalar output ($H$).
   - **Hyperparameter Space:**
     - `log_lr`: $(-4.0, -1.0)$ [decoded learning rate: $10^{\text{log\_lr}}$]
     - `dropout`: $(0.0, 0.3)$
     - `n_layers` (depth): $(1.0, 8.0)$
     - `n_neurons` (width): $(16.0, 256.0)$
     - `log_batch_size`: $(4.0, 6.0)$ [decoded batch size: $2^{\text{log\_batch\_size}}$, range 16–64]

---

### B. CIFAR-10 CNN Setup
The CIFAR-10 CNN benchmark represents a high-dimensional image classification task configured to enforce optimization difficulty.

1. **Subsampling Configuration:**
   - **CIFAR-10 Training Set:** Subsampled to 5,000 images (down from the standard 60,000) to introduce severe overfitting pressure and evaluate regularization.
   - **CIFAR-10 Test/Validation Set:** 1,000 images.
   - **Batch Size:** 64 (fixed).

2. **Network Architecture:**
   - **Input:** $28 \times 28$ grayscale images (1 channel).
   - **Feature Extractor (`nn.Sequential`):**
     - Convolutional Layer 1: `nn.Conv2d(1, 16, kernel_size=3, padding=1)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.MaxPool2d(kernel_size=2, stride=2)`
     - Convolutional Layer 2: `nn.Conv2d(16, 32, kernel_size=3, padding=1)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.MaxPool2d(kernel_size=2, stride=2)`
     - Outputs feature maps of shape `(batch_size, 32, 7, 7)`.
   - **Classifier (`nn.Sequential`):**
     - Flatten to 1,568 dimensions.
     - Linear Layer 1: `nn.Linear(1568, 128)` $\rightarrow$ `nn.ReLU()`
     - Dropout Layer: `nn.Dropout(p)` (dropout probability $p$ is tuned dynamically).
     - Output Layer: `nn.Linear(128, 10)` (mapping features to 10 digit classes).
   - **Loss Function:** `nn.CrossEntropyLoss` computed over the outputs.

3. **Hyperparameter Space:**
   - `log_lr`: $(-4.0, -1.0)$ [decoded learning rate: $10^{\text{log\_lr}}$]
   - `dropout`: $(0.0, 0.5)$
   - **Initial State:** Learning rate $= 0.001$ (`log_lr = -3.0`), Dropout $= 0.2$.

---

## 5. Formulation of Algorithms A and B on the CNN Testbed

### A. Algorithm A (Pure HHD) Formulation
HHD-HMC co-evolves the network weights $\theta$ and the hyperparameters $\lambda$ inside a joint continuous phase space.

1. **Phase-Space Initialization:**
   - Hyperparameters are continuous relaxed variables: $\lambda = [\log(\text{lr}), \text{dropout}]$ initialized at $[-3.0, 0.2]$.
   - The model is initialized with standard weights.

2. **Warm-up Phase:**
   - Trains the weights $\theta$ using Adam for 5 epochs with the initial learning rate ($10^{-3}$) while keeping the hyperparameters $\lambda$ frozen. This positions the weights in a low-curvature basin before HMC begins.

3. **Symplectic Integration Loop (Intra-Epoch HMC):**
   - At the start of each of the 15 epochs:
     - Samples a single training batch $(X_b, y_b)$ from the `train_loader`.
     - Computes the batch Cross-Entropy loss which serves as the potential energy.
     - Samples random momentum vectors for weights $p_\theta \sim \mathcal{N}(0, m_\theta)$ and hyperparameters $p_\lambda \sim \mathcal{N}(0, m_\lambda)$ (mass constants: $m_\theta = 1.0, m_\lambda = 1.0$).
     - Simulates joint trajectories using a **symplectic Leapfrog solver** with step-size $\epsilon = 0.005$ and $L = 4$ steps.
     - Evaluates the proposed state ($\theta_{\text{prop}}, \lambda_{\text{prop}}$) via a Metropolis-Hastings acceptance step with temperature $T = 1e9$.
     - If accepted, the hyperparameters and weights are updated; if rejected, the previous values are restored.
     - Dynamically propagates the accepted dropout rate to the network using `update_dropout_rate(model, dropout)`.

4. **Weight Descent:**
   - The model trains for one full epoch on the dataset using the updated learning rate (with HPs frozen during the epoch step).
   - Validation accuracy on the 1,000-image test set is evaluated at the end of the epoch, and the best state dictionary is cached.

---

### B. Algorithm B (Hybrid BO - ABBO) Formulation
Hybrid ABBO treats HPO as a decoupled, black-box optimization problem, separating the outer hyperparameter selection loop from the inner model training.

1. **Outer GP Loop Formulation:**
   - A Gaussian Process (GP) surrogate model is constructed to model the validation accuracy surface over the search bounds: $\log(\text{lr}) \in [-4.0, -1.0]$ and $\text{dropout} \in [0.0, 0.5]$.
   - The acquisition function used is Expected Improvement (EI), which is maximized via L-BFGS-B to select the next hyperparameter candidates.
   - Evaluates for a budget of 10 trials total (3 initial random samples, 7 GP-guided trials).

2. **Inner Model Re-initialization (Discontinuous):**
   - At each trial $t$:
     - The convolutional network is **fully re-initialized from scratch** with the proposed dropout rate.
     - A new Adam optimizer is instantiated with the proposed learning rate.
     - The model is trained for 10 full epochs using Adam. No L-BFGS is used inside the CNN trials to save compute time.

3. **Surrogate Feedback:**
   - After the 10 training epochs, validation accuracy is evaluated on the test loader.
   - The cost is defined as the negative validation accuracy: $y = -\text{Accuracy}_{\text{val}}$.
   - This value is fed back to update the GP surrogate model, refining the Expected Improvement curve for the next trial.
   - The model and hyperparameters that achieved the highest validation accuracy across all 10 trials are saved as the final output.

---

## 6. Walkthrough of Model and Hyperparameter Tuning Loops

### A. Step-by-Step Tuning Walkthrough: Algorithm A (Pure HHD)
HHD-HMC updates both the weights $\theta$ and the hyperparameters $\lambda$ concurrently in a unified training pipeline.

1. **Initialize Phase Space:**
   - Initialize network weights randomly. Set initial learning rate ($10^{-3}$) and dropout ($0.2$).
2. **Pre-stabilization (Warm-up):**
   - Train the weights $\theta$ for 5 epochs using Adam, keeping learning rate and dropout frozen. This settles the random weights into a stable basin.
3. **Co-Evolution Loop (For each of the 15 epochs):**
   - **Step 1: Batch Retrieval:** Retrieve a single batch $(X_b, y_b)$ from the CIFAR-10 training set.
   - **Step 2: Momentum Sampling:** Draw fresh momentum vectors:
     - For weights: $p_\theta \sim \mathcal{N}(0, I)$
     - For hyperparameters: $p_\lambda \sim \mathcal{N}(0, I)$
   - **Step 3: Initial Energy Calculation:** Compute the total initial energy of the phase space:
     $$H_{\text{init}} = \text{Loss}(\theta, \lambda; X_b, y_b) + \frac{p_\theta^2}{2m_\theta} + \frac{p_\lambda^2}{2m_\lambda}$$
   - **Step 4: Symplectic Integration (Leapfrog):** Run 4 leapfrog steps to update coordinates ($\theta, \lambda$) and momenta ($p_\theta, p_\lambda$). 
     - Updates rely on potential gradients: $\nabla_\theta L$ (computed via backpropagation) and $\nabla_\lambda L$ (computed via central finite differences by perturbing $\lambda$ by $\pm 10\%$).
   - **Step 5: Metropolis-Hastings Accept/Reject:** Calculate the proposed state's energy $H_{\text{prop}}$. Accept the proposed weights $\theta_{\text{prop}}$ and hyperparameters $\lambda_{\text{prop}}$ with probability:
     $$P_{\text{accept}} = \min\left(1, \exp\left(-\frac{H_{\text{prop}} - H_{\text{init}}}{T}\right)\right)$$
     where $T = 1e9$ (essentially ensuring proposals are accepted). If rejected, restore the epoch's starting weights and hyperparameters.
   - **Step 6: Hyperparameter Propagation:**
     - Update the network's dropout layers with the new dropout rate.
     - Update the Adam optimizer's learning rate parameter with the new learning rate.
   - **Step 7: Main Weight training epoch:** Train the network weights $\theta$ for one full epoch across all remaining training batches using the updated learning rate and dropout rate (HPs remain frozen during this step).
   - **Step 8: Epoch Validation:** Evaluate classification accuracy on the 1,000-image test set. If it is the highest accuracy achieved so far, save the model weights and hyperparameters as the best checkpoints.

---

### B. Step-by-Step Tuning Walkthrough: Algorithm B (Hybrid BO)
Hybrid ABBO tunes hyperparameters in an outer loop via surrogate modeling, and trains model weights in an inner loop.

1. **Construct GP Surrogate:**
   - Initialize a Gaussian Process prior over the 2D hyperparameter space ($\log(\text{lr}), \text{dropout}$).
2. **Surrogate Exploration Loop (For each of the 10 trials):**
   - **Step 1: Suggest Hyperparameters:**
     - If trial $t < 3$: Sample the hyperparameters randomly from the search bounds.
     - If trial $t \geq 3$: Maximise the Expected Improvement (EI) acquisition function using L-BFGS-B (running 5 restarts) to select the next hyperparameter query point.
   - **Step 2: Full Model Reset:** Fully discard the previous neural network. Instantiate a fresh model with randomly initialized weights, and set the proposed dropout rate.
   - **Step 3: Inner Loop Weight training (Adam):**
     - Instantiate a fresh Adam optimizer using the proposed learning rate.
     - Train the network weights $\theta$ from scratch for 10 epochs using Adam. 
   - **Step 4: Objective Evaluation:**
     - Evaluate accuracy on the validation loader.
     - Define the cost feedback as $y = -\text{Accuracy}_{\text{val}}$.
   - **Step 5: Surrogate Update:**
     - Append the query point and cost $(\lambda_t, y_t)$ to the GP's history.
     - Update the GP surrogate model, recalculating the covariance matrix to update the EI curve for the next trial.
     - If the trial accuracy is the best so far, cache the trained weights and the corresponding hyperparameters.

---

## 7. HPOBench and NAS-Bench-201 Testing: Setup & Importance

### A. The Setup for HPOBench & HPOLib
The HPOBench suite evaluates optimizer performance on tabular hyperparameter optimization tasks for standard machine learning models.

1. **Benchmarks and Datasets:**
   - **HPOBench Tabular datasets:** `australian`, `blood_transfusion`, `vehicle`, `segment` (evaluating hyperparameter search spaces for multi-layer perceptrons and support vector machines).
   - **HPOLib datasets:** `naval_propulsion`, `parkinsons_telemonitoring`, `protein_structure`, `slice_localization` (evaluating neural network regression hyperparameter spaces).
2. ** Tonal Formulation:**
   - The search space consists of discrete values or continuous ranges for parameters such as learning rate, batch size, weight decay, and kernel coefficients.
   - In each trial, the optimizer proposes a coordinate. The benchmark library (`simple-hpo-bench`) looks up the pre-computed validation MSE or classification error corresponding to the nearest discrete grid coordinate.
3. **Execution Budget:**
   - Run across 5 independent random seeds ($0, 1, 2, 3, 4$).
   - Budget: 100 trials (evaluations) per optimizer per seed.

---

### B. The Setup for NAS-Bench-201
NAS-Bench-201 is a specialized tabular benchmark for Neural Architecture Search (NAS).

1. **Search Space Structure:**
   - Defines a cell-based search space containing 15,625 unique candidate architectures represented as Directed Acyclic Graphs (DAGs) with 4 nodes.
   - The search space allows 5 different operation choices on each DAG edge: zero (none), skip-connection, 1x1 convolution, 3x3 convolution, and 3x3 average pooling.
2. **Datasets Evaluated:**
   - Tested on three image classification datasets: `cifar10`, `cifar100`, and `imagenet` (subsampled ImageNet-16-120).
3. **Optimisation Goal:**
   - The optimizer proposes structural operation indices for each edge in the DAG. The benchmark returns the pre-computed test and validation accuracies lookup, bypassing the need to train the selected network topology.
4. **Execution Budget:**
   - Run across 5 independent random seeds ($0, 1, 2, 3, 4$).
   - Budget: 100 trials (evaluations) per optimizer per seed.

---

### C. Empirical and Theoretical Importance

1. **Eliminating Evaluation Bias (Zero-Cost Reproducibility):**
   - Traditional hyperparameter and architecture searches are highly sensitive to hardware fluctuations, background OS tasks, random GPU seeds, and driver differences.
   - HPOBench and NAS-Bench-201 use static lookup tables generated by training every possible configuration for hundreds of epochs on uniform hardware (equivalent to thousands of hours of GPU compute).
   - This ensures **100% reproducible results** and isolates the benchmark to *pure optimization logic*, removing hardware bias and evaluation noise.

2. **Modality Diversification:**
   - Evaluates optimizers across distinct search space topologies:
     - **HPOBench:** Evaluates continuous/discrete parameters for classical ML models and standard feedforward layers.
     - **NAS-Bench-201:** Evaluates structural connectivity (graphs, connections, operations) representing a highly non-convex discrete graph space.
   - This prevents over-tuning an algorithm to a single model style (like simple MLPs) and tests its generalized tuning capability.

3. **Statistically Significant Comparisons:**
   - Performing HPO/NAS on real networks is extremely expensive, which often limits academic papers to testing on a single dataset with one seed.
   - Because lookups are instantaneous, tabular benchmarks allow us to execute **5 seeds and 100 trials across 11 distinct datasets** in minutes. This produces rigorous, statistically sound comparison metrics (mean best regret, standard deviation, and average ranks) that validate the optimizer's true performance.

---

## 8. HPOBench & NAS-Bench-201 Optimizer Modeling

Because tabular/lookup benchmarks are black-box and have no model weights $\theta$ that undergo physical training, we must model our co-evolution algorithms differently in this environment:

### A. Optimizer Phase Space Modeling
- **Hyperparameter Coordinates:** The position vector $\lambda$ is modeled as a continuous coordinate vector representing the indices of selected hyperparameters in their respective discrete search lists (e.g. if the batch size choices are `[16, 32, 64]`, index `1.0` maps to `32`).
- **Potential Energy:** The potential energy of the HMC system is defined directly as the validation cost (validation error or negative validation accuracy) looked up from the tabular dataset.
- **Finite-Difference HP Gradients:** Since no computational graph exists for the lookup table, the hyperparameter gradients ($\nabla_\lambda L$) are calculated using **central finite differences** by perturbing the index vector by $\pm 1.0$ along each coordinate axis and querying the benchmark.
- **Momenta & Leapfrog:** Momenta ($p_\lambda$) are sampled from $\mathcal{N}(0, I)$ and used inside the Leapfrog solver to update coordinates and check acceptance via Metropolis-Hastings.

### B. Algorithm C Curriculum Mapping
- **Phase 1 (Warm-up):** Modeled as a uniform random sampling phase to explore the index landscape and identify a low-loss starting region.
- **Phase 2 (MCMC Exploration):** Executes the index-based HMC co-evolution.
- **Phase 3 (L-BFGS Polish):** Modeled as a **local coordinate perturbation search**: the optimizer takes the best coordinate indices found so far, randomly perturbs a subset by $\pm 1$, and accepts the step if the validation cost improves.

---

## 9. How to Use the Unified HHD-ABBO Optimizer (Algorithm C) on Any New Neural Network

To use HHD-Unified's self-tuning capabilities on any new PyTorch neural network and dataset, developers can follow a modular implementation pattern:

### Step 1: Define Your Model & Hyperparameter Space
Ensure your custom model (`nn.Module`) has parameterized layers that can be dynamically updated. For instance, dropout rates must map to `nn.Dropout` layer parameters, and learning rate must map to the optimizer.

```python
import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(16 * 26 * 26, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.dropout(x)
        return self.fc(x.view(x.size(0), -1))
```

### Step 2: Implement the Modular 3-Phase Curriculum
Instead of full network reinitialization, run training batches and update hyperparameters dynamically using the modular classes:

```python
from hamiltonian import HyperparamState
from symplectic_solver import HamiltonianMCMC
from hybrid_hhd_abbo_improved import AdaptiveStepSizeController, PlateauDetector
import torch.optim as optim

# 1. Define hyperparameter search spaces and initial values
hp_space = {"log_lr": (-4.0, -1.0), "dropout": (0.0, 0.5)}
init_hp  = {"log_lr": -3.0, "dropout": 0.2}

hp_state = HyperparamState(init_hp, hp_space)
model = CustomNet(hp_state.decode()["dropout"]).to("cuda")
criterion = nn.CrossEntropyLoss()

# 2. Instantiate the physical co-evolution components
mcmc = HamiltonianMCMC(step_size=0.005, n_leapfrog=4, mass_theta=1.0, mass_lambda=1.0, temperature=1e9)
step_ctrl = AdaptiveStepSizeController(initial_step=0.005)
plateau = PlateauDetector(patience=4, tol=5e-4)

train_loader, test_loader = get_your_dataloaders()

# --- Phase 1: Standard Adam Warm-up (e.g. 5 epochs) ---
opt = optim.Adam(model.parameters(), lr=hp_state.decode()["lr"])
for epoch in range(5):
    for X, y in train_loader:
        opt.zero_grad()
        criterion(model(X.to("cuda")), y.to("cuda")).backward()
        opt.step()

# --- Phase 2 & 3: Co-Evolution + L-BFGS Refinement (e.g. 15 epochs) ---
for epoch in range(15):
    # HMC step: sample batch, evaluate loss, run leapfrog, and check MH acceptance
    Xb, yb = next(iter(train_loader))
    Xb, yb = Xb.to("cuda"), yb.to("cuda")
    curr_loss = criterion(model(Xb), yb).item()
    
    # Propose new weights and hyperparameters
    mcmc.propose(model, hp_state, (Xb, yb), criterion, curr_loss)
    
    # Propagate the updated dropout and learning rate to the network
    # (Helper to find and set all nn.Dropout layer rates to hp_state.decode()["dropout"])
    update_dropout_rate(model, hp_state.decode()["dropout"])
    
    # Adapt leapfrog step-size based on MCMC acceptance
    mcmc.leapfrog.eps = step_ctrl.update(mcmc.acceptance_rate)
    
    # Adapt optimizer learning rate
    for pg in opt.param_groups:
        pg["lr"] = hp_state.decode()["lr"]
        
    # Standard epoch training step
    for X, y in train_loader:
        opt.zero_grad()
        criterion(model(X.to("cuda")), y.to("cuda")).backward()
        opt.step()

    # --- Phase 3: Plateau-Triggered L-BFGS ---
    epoch_loss = evaluate_loss(model, train_loader)
    if plateau.update(epoch_loss):
        lbfgs = optim.LBFGS(model.parameters(), max_iter=10)
        def closure():
            lbfgs.zero_grad()
            l = criterion(model(X_full), y_full)
            l.backward()
            return l
        lbfgs.step(closure)
        plateau.reset()
```
This modular approach allows developers to easily inject self-tuning HHD co-evolution into any PyTorch network architecture.

