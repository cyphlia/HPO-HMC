"""Quick timing test: Method A seed=0 with reduced config."""
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

print(f"Train subset:  {config.CNN_TRAIN_SUBSET}")
print(f"Test subset:   {config.CNN_TEST_SUBSET}")
print(f"Warmup epochs: {config.CNN_WARMUP_EPOCHS}")
print(f"HMC epochs:    {config.CNN_HMC_EPOCHS}")
print(f"BO trials:     {config.CNN_BO_TRIALS}")
print("-" * 40)

import torch
torch.set_num_threads(1)

from cnn_benchmark import run_method_a_cnn

t0 = time.time()
r = run_method_a_cnn(seed=0)
elapsed = time.time() - t0
print(f"\nMethod A seed=0 => acc={r['best_val_acc']:.2%}  time={elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Estimated total for 15 tasks: {elapsed*15/60:.0f} min")
