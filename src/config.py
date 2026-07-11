"""
Configuration for Hamiltonian Hyperparameter Dynamics (HHD).

Centralizes all physics constants, integration parameters, search spaces,
and training settings used across all three optimization methods.
"""

# --------------------------------------------------------------------------- #
#  Physical Constants (Hamiltonian system)
# --------------------------------------------------------------------------- #
MASS_THETA  = 1.0      # Inertia for network-weight momentum
MASS_LAMBDA = 5.0      # Inertia for hyperparameter momentum (higher = slower/more conservative HP movement)

# --------------------------------------------------------------------------- #
#  Leapfrog / HMC Integration
# --------------------------------------------------------------------------- #
STEP_SIZE        = 0.003    # eps: leapfrog step size (smaller = higher acceptance)
N_LEAPFROG_STEPS = 3        # L:   leapfrog steps per HMC proposal
TEMPERATURE      = 1e9      # T:   Boltzmann temperature (1e9 = always-accept for optimization)

# --------------------------------------------------------------------------- #
#  Training Defaults
# --------------------------------------------------------------------------- #
N_EPOCHS        = 80
N_WARMUP_EPOCHS = 30
N_SAMPLES       = 2500
SEED            = 101

# --------------------------------------------------------------------------- #
#  Harmonic Oscillator (ground-truth problem)
#  H(q,p) = p^2/(2m) + (1/2)k*q^2
# --------------------------------------------------------------------------- #
MASS_M   = 1.0   # particle mass
SPRING_K = 1.0   # spring constant

# --------------------------------------------------------------------------- #
#  Hyperparameter Search Space (continuous relaxations)
# --------------------------------------------------------------------------- #
HYPERPARAM_SPACE = {
    "log_lr"         : (-4.0, -1.0),     # log10(lr): 1e-4 to 1e-1
    "dropout"        : (0.0,  0.3),      # dropout probability
    "n_layers"       : (1.0,  8.0),      # continuous relaxation of depth
    "n_neurons"      : (16.0, 256.0),    # continuous relaxation of width
    "log_batch_size" : (4.0,  6.0),      # log2(bs): 16 to 64
}

# --------------------------------------------------------------------------- #
#  Initial Hyperparameters
# --------------------------------------------------------------------------- #
INIT_HYPERPARAMS = {
    "log_lr"         : -3.0,     # lr = 0.001
    "dropout"        : 0.1,
    "n_layers"       : 5.0,
    "n_neurons"      : 128.0,
    "log_batch_size" : 5.0,      # batch_size = 32
}

# --------------------------------------------------------------------------- #
#  CNN Benchmark Settings (CIFAR-10, Phase 5 SIAM revision)
# --------------------------------------------------------------------------- #
CNN_HP_SPACE = {
    "log_lr"         : (-4.0, -1.0),       # lr: 1e-4 to 1e-1
    "dropout"        : (0.0,  0.5),        # dropout probability
    "log_wd"         : (-6.0, -2.0),       # weight decay: 1e-6 to 1e-2
    "log_batch_size" : (5.0,  8.0),        # batch size: 32 to 256
}

CNN_INIT_HP = {
    "log_lr"         : -3.0,
    "dropout"        : 0.2,
    "log_wd"         : -4.0,
    "log_batch_size" : 6.0,   # batch_size = 64
}

CNN_TRAIN_SUBSET  = 400    # CIFAR-10 subset for fast CPU execution
CNN_TEST_SUBSET   = 200    # CIFAR-10 subset for fast CPU evaluation
CNN_WARMUP_EPOCHS = 1      # Adam warmup
CNN_HMC_EPOCHS    = 4      # HMC co-evolution epochs
CNN_BO_TRIALS     = 3      # Bayesian optimization trials
CNN_BATCH_SIZE    = 64

# --------------------------------------------------------------------------- #
#  HPOBench / HPOLib / NAS-Bench-201 Benchmark Settings
# --------------------------------------------------------------------------- #
HPOBENCH_TRIALS   = 100          # evaluation budget per optimizer per seed
HPOBENCH_SEEDS    = [0, 1, 2, 3, 4]

HPOBENCH_DATASETS  = ["australian", "blood_transfusion", "vehicle", "segment"]
HPOLIB_DATASETS    = ["naval_propulsion", "parkinsons_telemonitoring",
                      "protein_structure", "slice_localization"]
NASBENCH201_DATASETS = ["cifar10", "cifar100", "imagenet"]

HPOBENCH_OPTIMIZERS = [
    "RandomSearch", "OptunaTPE",
    "MethodA_HHD", "MethodB_ABBO", "MethodC_Unified",
    "BOHB", "SMAC3",
]

HPOBENCH_RESULTS_DIR = "results_hpobench"

# --------------------------------------------------------------------------- #
#  I/O Directories
# --------------------------------------------------------------------------- #
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
