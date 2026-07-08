"""Compare all auto_LiRPA verification methods for the inverted pendulum, 5 seeds.

Logs per-seed violation counts and timing to results/pendulum/pendulum_compare.csv.
Hardcoded IBP is omitted: the pendulum drift is nonlinear (sin + neural policy),
so the GBM-specific linear_drift_dot_grad_bounds doesn't apply.
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

from rl_agent import TanhPolicy
import controlled_sde
import stochastic_rsa as rsa

rl_policy_net = TanhPolicy(2, 1, 64, device=device)
rl_policy_net.load_state_dict(torch.load(
    "rl_agent/pendulum_policy.pt",
    map_location=device,
    weights_only=True,
))
rl_policy_net.requires_grad_(False)

sde = controlled_sde.InvertedPendulum(rl_policy_net)

global_bounds  = np.array([[[-20.0, -2*np.pi], [20.0, 2*np.pi]]])
initial_bounds = np.array([[[-1.0, 3/4*np.pi], [1.0, 5/4*np.pi]]])
target_bounds  = np.array([[[-4.0, -np.pi/2], [4.0, np.pi/2]]])
unsafe_bounds  = np.array([
    [[-20.0, -2*np.pi], [-10.0, -3/2*np.pi]],
    [[10.0, 3/2*np.pi], [20.0, 2*np.pi]],
])

interest_set = rsa.AABBSet(global_bounds, device)
initial_set  = rsa.AABBSet(initial_bounds, device)
target_set   = rsa.AABBSet(target_bounds, device)
unsafe_set   = rsa.AABBSet(unsafe_bounds, device)
spec = rsa.Specification(interest_set, initial_set, unsafe_set, target_set, 0.9, 0.9)

npr.seed(1)
seeds = npr.randint(1, int(1e5), size=(5,)).tolist()

# (label, use_jacobian_verifier, bound_method, use_pendulum_hardcoded_ibp)
configs = [
    ('analytical_IBP',       False, 'IBP',       False),
    ('DJ_IBP',               True,  'IBP',       False),
    ('hardcoded_IBP',        False, 'IBP',       True),
    ('DJ_CROWN-IBP',         True,  'CROWN-IBP', False),
    ('analytical_CROWN-IBP', False, 'CROWN-IBP', False),
]

with open('results/pendulum/pendulum_compare.csv', 'w', newline='') as f:
    csv.writer(f).writerow(
        ['seed', 'method', 'checkpoint_epoch', 'violations', 'converged', 'elapsed_s']
    )

for seed in seeds:
    for label, use_jop, method, use_hc_pend in configs:
        print(f"\n{'='*60}")
        print(f"  seed={seed}  {label}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        npr.seed(seed)

        net  = rsa.CertificateModule(device=device)
        cert = rsa.SupermartingaleCertificate(sde, spec, net, device)

        t0 = time.time()
        converged, epoch, violation_history = cert.train(
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
            use_hardcoded_ibp=False,
            use_pendulum_hardcoded_ibp=use_hc_pend,
        )
        elapsed = time.time() - t0

        print(f"\nresult: converged={converged}, epoch={epoch}, time={elapsed:.1f}s")

        with open('results/pendulum/pendulum_compare.csv', 'a', newline='') as f:
            w = csv.writer(f)
            for chk_epoch, n_viol in violation_history:
                w.writerow([seed, label, chk_epoch, n_viol, '', ''])
            w.writerow([seed, label, epoch, '', converged, f'{elapsed:.1f}'])
