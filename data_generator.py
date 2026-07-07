"""
Data Generator - Hamiltonian Systems.

Generates phase-space data for multiple Hamiltonian systems:

  1. Simple Harmonic Oscillator (2D):
     H(q, p) = p^2 / (2m) + (1/2) k q^2

  2. Henon-Heiles System (4D, non-integrable, chaotic):
     H(q1,q2,p1,p2) = (p1^2+p2^2)/2 + (q1^2+q2^2)/2 + q1^2*q2 - q2^3/3

  3. Double-Well Potential (2D, bimodal):
     H(q, p) = p^2/2 + (q^2 - 1)^2

  4. Kepler Two-Body Problem (4D, singular potential) [optional]:
     H(q1,q2,p1,p2) = (p1^2+p2^2)/2 - 1/sqrt(q1^2+q2^2)

Returns train/val DataLoaders and mesh grids for visualization.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


def generate_hamiltonian_data(
    n_samples: int = 2500,
    batch_size: int = 32,
    m: float = 1.0,
    k: float = 1.0,
    noise_std: float = 0.05,
    seed: int = 101,
) -> Tuple[DataLoader, DataLoader, Tuple]:
    """
    Sample random (q, p) pairs from phase space and compute H.

    Parameters
    ----------
    n_samples  : total number of (q, p) data points
    batch_size : DataLoader batch size
    m          : particle mass
    k          : spring constant
    noise_std  : std of additive Gaussian noise on H
    seed       : reproducibility seed

    Returns
    -------
    train_loader, val_loader, (q_mesh, p_mesh, H_true_mesh)
    """
    np.random.seed(seed)

    # Phase-space sampling
    q = np.random.uniform(-4, 4, (n_samples, 1)).astype(np.float32)
    p = np.random.uniform(-4, 4, (n_samples, 1)).astype(np.float32)

    H = (p ** 2) / (2 * m) + 0.5 * k * q ** 2
    if noise_std > 0:
        H += np.random.normal(0, noise_std, H.shape).astype(np.float32)

    X = np.hstack([q, p])  # shape (N, 2)

    # 80/20 train-val split
    split = int(0.8 * n_samples)
    X_tr, X_va = X[:split], X[split:]
    H_tr, H_va = H[:split], H[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(H_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(H_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Visualization mesh (50x50)
    qs, ps = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
    q_mesh, p_mesh = np.meshgrid(qs, ps)
    H_mesh = (p_mesh ** 2) / (2 * m) + 0.5 * k * q_mesh ** 2

    return train_loader, val_loader, (q_mesh, p_mesh, H_mesh)


# --------------------------------------------------------------------------- #
#  Henon-Heiles System (4D phase space, non-integrable)
# --------------------------------------------------------------------------- #

def henon_heiles_hamiltonian(q1, q2, p1, p2):
    """
    Henon-Heiles Hamiltonian:
      H = (p1^2 + p2^2)/2 + (q1^2 + q2^2)/2 + q1^2*q2 - q2^3/3

    This is a 2-degree-of-freedom non-integrable system that exhibits a
    transition from regular to chaotic motion. Standard benchmark in the
    Hamiltonian neural network literature (Greydanus 2019, Offen & Ober-Blobaum 2022).
    """
    kinetic = 0.5 * (p1**2 + p2**2)
    harmonic = 0.5 * (q1**2 + q2**2)
    coupling = q1**2 * q2 - q2**3 / 3.0
    return kinetic + harmonic + coupling


def generate_henon_heiles_data(
    n_samples: int = 2000,
    batch_size: int = 32,
    noise_std: float = 0.05,
    seed: int = 101,
    range_limit: float = 4.0,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Generate phase-space data for the Henon-Heiles system.

    Samples from [-range_limit, range_limit]^4 in (q1, q2, p1, p2) space.

    Parameters
    ----------
    n_samples   : total number of data points
    batch_size  : DataLoader batch size
    noise_std   : std of additive Gaussian noise on H
    seed        : reproducibility seed
    range_limit : sampling range for each phase-space coordinate

    Returns
    -------
    train_loader, val_loader, metadata_dict
    """
    np.random.seed(seed)

    q1 = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)
    q2 = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)
    p1 = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)
    p2 = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)

    H = henon_heiles_hamiltonian(q1, q2, p1, p2)
    if noise_std > 0:
        H += np.random.normal(0, noise_std, H.shape).astype(np.float32)

    X = np.hstack([q1, q2, p1, p2])  # shape (N, 4)

    # 80/20 train-val split
    split = int(0.8 * n_samples)
    X_tr, X_va = X[:split], X[split:]
    H_tr, H_va = H[:split], H[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(H_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(H_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Visualization: 2D slice at p1=0, p2=0
    n_grid = 50
    qs = np.linspace(-range_limit, range_limit, n_grid)
    q1_mesh, q2_mesh = np.meshgrid(qs, qs)
    H_mesh = henon_heiles_hamiltonian(
        q1_mesh, q2_mesh,
        np.zeros_like(q1_mesh), np.zeros_like(q2_mesh)
    )

    metadata = {
        "system": "henon_heiles",
        "input_dim": 4,
        "q1_mesh": q1_mesh,
        "q2_mesh": q2_mesh,
        "H_mesh": H_mesh,
        "n_samples": n_samples,
        "range_limit": range_limit,
    }

    return train_loader, val_loader, metadata


# --------------------------------------------------------------------------- #
#  Double-Well Potential (2D phase space, bimodal)
# --------------------------------------------------------------------------- #

def double_well_hamiltonian(q, p):
    """
    Double-well potential:
      V(q) = (q^2 - 1)^2
      H(q, p) = p^2/2 + (q^2 - 1)^2

    Two-well landscape with a saddle point at q=0. Tests whether the
    optimiser correctly learns a bimodal energy surface. Directly relevant
    to the argument that Method C handles non-convex landscapes better
    than Method B.
    """
    kinetic = 0.5 * p**2
    potential = (q**2 - 1)**2
    return kinetic + potential


def generate_double_well_data(
    n_samples: int = 2000,
    batch_size: int = 32,
    noise_std: float = 0.05,
    seed: int = 101,
    range_limit: float = 4.0,
) -> Tuple[DataLoader, DataLoader, Tuple]:
    """
    Generate phase-space data for the double-well potential.

    Parameters
    ----------
    n_samples   : total number of data points
    batch_size  : DataLoader batch size
    noise_std   : std of additive Gaussian noise on H
    seed        : reproducibility seed
    range_limit : sampling range

    Returns
    -------
    train_loader, val_loader, (q_mesh, p_mesh, H_true_mesh)
    """
    np.random.seed(seed)

    q = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)
    p = np.random.uniform(-range_limit, range_limit, (n_samples, 1)).astype(np.float32)

    H = double_well_hamiltonian(q, p)
    if noise_std > 0:
        H += np.random.normal(0, noise_std, H.shape).astype(np.float32)

    X = np.hstack([q, p])  # shape (N, 2)

    # 80/20 train-val split
    split = int(0.8 * n_samples)
    X_tr, X_va = X[:split], X[split:]
    H_tr, H_va = H[:split], H[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(H_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(H_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Visualization mesh (50x50)
    n_grid = 50
    qs = np.linspace(-range_limit, range_limit, n_grid)
    ps = np.linspace(-range_limit, range_limit, n_grid)
    q_mesh, p_mesh = np.meshgrid(qs, ps)
    H_mesh = double_well_hamiltonian(q_mesh, p_mesh)

    return train_loader, val_loader, (q_mesh, p_mesh, H_mesh)


# --------------------------------------------------------------------------- #
#  Kepler Two-Body Problem (4D phase space, singular potential) [optional]
# --------------------------------------------------------------------------- #

def kepler_hamiltonian(q1, q2, p1, p2):
    """
    Kepler two-body problem:
      H = |p|^2/2 - 1/|q|
      H(q1,q2,p1,p2) = (p1^2+p2^2)/2 - 1/sqrt(q1^2+q2^2)

    Singular potential at origin. Tests robustness of leapfrog
    integration near the singularity.
    """
    kinetic = 0.5 * (p1**2 + p2**2)
    r = np.sqrt(q1**2 + q2**2 + 1e-8)  # regularize singularity
    potential = -1.0 / r
    return kinetic + potential


def generate_kepler_data(
    n_samples: int = 2000,
    batch_size: int = 32,
    noise_std: float = 0.05,
    seed: int = 101,
    range_limit: float = 4.0,
    min_radius: float = 0.3,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Generate phase-space data for the Kepler two-body problem.

    Avoids sampling too close to the singularity at the origin
    by rejecting samples with |q| < min_radius.

    Parameters
    ----------
    n_samples   : total number of data points
    batch_size  : DataLoader batch size
    noise_std   : std of additive Gaussian noise on H
    seed        : reproducibility seed
    range_limit : sampling range
    min_radius  : minimum distance from origin (avoids singularity)

    Returns
    -------
    train_loader, val_loader, metadata_dict
    """
    np.random.seed(seed)

    # Over-sample and reject points too close to origin
    X_list = []
    H_list = []
    while len(X_list) < n_samples:
        batch = max(n_samples * 2, 5000)
        q1 = np.random.uniform(-range_limit, range_limit, (batch, 1)).astype(np.float32)
        q2 = np.random.uniform(-range_limit, range_limit, (batch, 1)).astype(np.float32)
        p1 = np.random.uniform(-range_limit, range_limit, (batch, 1)).astype(np.float32)
        p2 = np.random.uniform(-range_limit, range_limit, (batch, 1)).astype(np.float32)

        r = np.sqrt(q1**2 + q2**2)
        mask = (r > min_radius).flatten()

        q1, q2, p1, p2 = q1[mask], q2[mask], p1[mask], p2[mask]
        H = kepler_hamiltonian(q1, q2, p1, p2)
        if noise_std > 0:
            H += np.random.normal(0, noise_std, H.shape).astype(np.float32)

        X_batch = np.hstack([q1, q2, p1, p2])
        for i in range(len(X_batch)):
            if len(X_list) >= n_samples:
                break
            X_list.append(X_batch[i])
            H_list.append(H[i])

    X = np.array(X_list, dtype=np.float32)
    H = np.array(H_list, dtype=np.float32).reshape(-1, 1)

    # 80/20 train-val split
    split = int(0.8 * n_samples)
    X_tr, X_va = X[:split], X[split:]
    H_tr, H_va = H[:split], H[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(H_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(H_va))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Visualization: 2D slice at p1=0, p2=0
    n_grid = 50
    qs = np.linspace(-range_limit, range_limit, n_grid)
    q1_mesh, q2_mesh = np.meshgrid(qs, qs)
    r_mesh = np.sqrt(q1_mesh**2 + q2_mesh**2)
    # Mask out singularity for visualization
    H_mesh = kepler_hamiltonian(
        q1_mesh, q2_mesh,
        np.zeros_like(q1_mesh), np.zeros_like(q2_mesh)
    )
    H_mesh = np.clip(H_mesh, -20, 20)  # clip extreme values near singularity

    metadata = {
        "system": "kepler",
        "input_dim": 4,
        "q1_mesh": q1_mesh,
        "q2_mesh": q2_mesh,
        "H_mesh": H_mesh,
        "n_samples": n_samples,
        "range_limit": range_limit,
    }

    return train_loader, val_loader, metadata

