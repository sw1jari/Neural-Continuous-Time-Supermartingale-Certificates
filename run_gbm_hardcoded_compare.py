"""Compare all GBM verification methods across 5 seeds.

Logs per-seed violation counts and timing to gbm_hardcoded_compare.csv.
Checkpoint rows are written immediately as each verification fires, so a
crash only loses the current training run rather than the whole experiment.
Completed (seed, method) pairs are skipped on restart.
"""

import csv
import os
import time

import numpy as np
import numpy.random as npr
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

import controlled_sde
import stochastic_rsa as rsa

global_bounds = np.array([[[-100.0, -100.0], [100.0, 100.0]]])
initial_bounds = np.array([[[45, -55], [55, -45]]])
target_bounds  = np.array([[[-25.0, -25.0], [25.0, 25.0]]])
unsafe_bounds  = np.array([[[-100.0, -100.0], [-80.0, 100.0]]])

interest_set = rsa.AABBSet(global_bounds, device)
initial_set  = rsa.AABBSet(initial_bounds, device)
target_set   = rsa.AABBSet(target_bounds, device)
unsafe_set   = rsa.AABBSet(unsafe_bounds, device)
spec = rsa.Specification(interest_set, initial_set, unsafe_set, target_set, 0.9, 0.9)

sde = controlled_sde.GBM()

# each config: (label, use_jacobian_verifier, bound_method, use_hardcoded_ibp)
all_configs = [
    ('analytical_IBP',       False, 'IBP',       False),
    ('DJ_IBP',               True,  'IBP',       False),
    ('hardcoded_IBP',        False, 'IBP',       True),
    ('DJ_CROWN-IBP',         True,  'CROWN-IBP', False),
    ('analytical_CROWN-IBP', False, 'CROWN-IBP', False),
]
# tighter subset for extra seeds: drop the analytical hessian methods since
# the question is whether DJ or MLP-IBP does better
focused_configs = [c for c in all_configs if 'analytical' not in c[0]]

# first 5 seeds match Neustroev et al.; 5 extra seeds use the focused subset
npr.seed(1)
original_seeds = npr.randint(1, int(1e5), size=(5,)).tolist()
npr.seed(2)
extra_seeds = npr.randint(1, int(1e5), size=(5,)).tolist()

seed_run_pairs = (
    [(s, all_configs)      for s in original_seeds] +
    [(s, focused_configs)  for s in extra_seeds]
)

CSV_PATH = 'results/gbm/gbm_hardcoded_compare.csv'
HEADER = ['seed', 'method', 'checkpoint_epoch', 'violations',
          'wall_s', 'converged', 'compile_s', 'total_elapsed_s']

# find which (seed, method) pairs already have a summary row (converged set)
# so we can skip them on restart
done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            if row['converged'] != '':
                done.add((int(row['seed']), row['method']))
else:
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(HEADER)

for seed, configs in seed_run_pairs:
    for label, use_jop, method, use_hc in configs:
        if (seed, label) in done:
            print(f"skipping seed={seed} {label} (already complete)")
            continue

        print(f"\n{'='*60}")
        print(f"  seed={seed}  {label}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        npr.seed(seed)

        net  = rsa.CertificateModule(device=device)
        cert = rsa.SupermartingaleCertificate(sde, spec, net, device)

        t0 = time.time()
        try:
            with open(CSV_PATH, 'a', newline='') as f:
                w = csv.writer(f)

                def on_checkpoint(chk_epoch, n_viol, wall_s):
                    w.writerow([seed, label, chk_epoch, n_viol,
                                f'{wall_s:.3f}', '', '', ''])
                    f.flush()

                converged, epoch, _, compile_s = cert.train(
                    n_epochs=20000,
                    verify_every_n=1000,
                    verifier_mesh_size=200,
                    zeta=1.0,
                    regularizer_lambda=1e-1,
                    verification_slack=4,
                    max_depth=2,
                    time_limit_s=600,
                    bound_method=method,
                    use_jacobian_verifier=use_jop,
                    use_hardcoded_ibp=use_hc,
                    checkpoint_callback=on_checkpoint,
                )
                elapsed = time.time() - t0

                # summary row: compile_s is measured separately so the plotter can
                # show "without compilation" as wall_s - compile_s for any checkpoint
                w.writerow([seed, label, epoch, '', '', converged,
                            f'{compile_s:.3f}', f'{elapsed:.1f}'])

            print(f"\nresult: converged={converged}, epoch={epoch}, "
                  f"time={elapsed:.1f}s, compile={compile_s:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\ncrashed: seed={seed} {label} after {elapsed:.1f}s: {e}")
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([seed, label, -1, '', '', f'error:{e}',
                                        '', f'{elapsed:.1f}'])
