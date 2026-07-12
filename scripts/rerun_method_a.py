import json, time, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation")))
import numpy as np, torch

import config
from train_hamiltonian import HamiltonianTrainer
from evaluate import load_model, predict_landscape
from data_generator import generate_hamiltonian_data

results = []
for seed in [0, 1, 2, 3, 4]:
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    tr = HamiltonianTrainer(
        hyperparam_space=config.HYPERPARAM_SPACE,
        init_hyperparams=config.INIT_HYPERPARAMS,
        step_size=config.STEP_SIZE,
        n_leapfrog=config.N_LEAPFROG_STEPS,
        temperature=config.TEMPERATURE,
        momentum_refresh=1.0,   # keep off by default; isolate the bugfix + checkpointing
    )
    tr.train(n_samples=config.N_SAMPLES, n_warmup=config.N_WARMUP_EPOCHS, n_hamilton=config.N_EPOCHS)
    dt = time.time() - t0

    _, _, mesh = generate_hamiltonian_data(n_samples=500)
    q_mesh, p_mesh, H_true = mesh
    hp = tr.hp_state.decode()
    model = tr.model
    model.eval()
    H_pred = predict_landscape(model, q_mesh, p_mesh)
    residual = np.abs(H_pred - H_true)
    mae = float(residual.mean())
    rmse = float(np.sqrt(((H_pred - H_true) ** 2).mean()))
    r2 = float(1 - ((H_pred - H_true) ** 2).sum() / ((H_true - H_true.mean()) ** 2).sum())
    best_val = min(tr.history["val_loss"])

    print(f"seed={seed}  best_val={best_val:.5f}  mae={mae:.4f}  rmse={rmse:.4f}  r2={r2:.4f}  time={dt:.1f}s")
    results.append(dict(seed=seed, best_val_loss=best_val, mae=mae, rmse=rmse, r2=r2, train_time=dt))

arr = lambda k: np.array([r[k] for r in results])
print()
print(f"MEAN best_val = {arr('best_val_loss').mean():.5f} +/- {arr('best_val_loss').std():.5f}")
print(f"MEAN r2       = {arr('r2').mean():.5f} +/- {arr('r2').std():.5f}")
print(f"MEAN mae      = {arr('mae').mean():.5f} +/- {arr('mae').std():.5f}")
print(f"MEAN time     = {arr('train_time').mean():.2f}s")

json.dump(results, open("method_a_fixed_results.json", "w"), indent=2)
