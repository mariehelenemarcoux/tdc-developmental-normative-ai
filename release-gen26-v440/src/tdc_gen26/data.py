from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = [
    "evidence_support", "contradiction", "evidence_independence", "temporal_novelty",
    "T_E", "T_N", "T_S",
    "local_uncertainty", "meso_uncertainty", "global_uncertainty",
    "decision_flip_probability", "corrective_harmful_separability",
    "irreversibility", "role_symmetry", "legitimacy", "correctability",
    "long_horizon_risk", "current_authority",
    "meta_epistemic_confidence", "meta_normative_confidence", "meta_resolution_confidence",
    "strategic_conflict", "observation_inference_gap", "severe_harm",
]


class SyntheticTDCDataset(Dataset):
    """Deterministic synthetic imitation task for smoke training and regression tests."""

    def __init__(self, n: int = 2048, seed: int = 440, input_dim: int = 24) -> None:
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(f"input_dim must be {len(FEATURE_NAMES)} for this dataset")
        rng = np.random.default_rng(seed)
        x = rng.beta(2.0, 2.0, size=(n, input_dim)).astype(np.float32)

        # Structured correlations.
        x[:, 10] = np.clip(0.55 * x[:, 10] + 0.35 * np.abs(x[:, 7] - x[:, 9]) + rng.normal(0, .05, n), 0, 1)
        x[:, 11] = np.clip(0.65 * x[:, 11] + 0.20 * x[:, 2] + rng.normal(0, .05, n), 0, 1)
        x[:, 22] = np.clip(0.60 * x[:, 22] + 0.25 * x[:, 2], 0, 1)

        evidence_support=x[:,0]; contradiction=x[:,1]; independence=x[:,2]
        local_u=x[:,7]; meso_u=x[:,8]; global_u=x[:,9]
        pflip=x[:,10]; sep=x[:,11]
        irreversibility=x[:,12]; role=x[:,13]; legitimacy=x[:,14]; correctability=x[:,15]
        longrisk=x[:,16]; authority=x[:,17]; conflict=x[:,21]; obs_gap=x[:,22]; severe=x[:,23]

        core_risk=((irreversibility*severe>=.58)|((irreversibility>=.8)&(severe>=.55))).astype(np.float32)
        readiness=.30*role+.35*legitimacy+.20*correctability+.15*obs_gap

        action=np.zeros(n,dtype=np.int64)
        action[core_risk>0]=1
        action[(core_risk>0)&(correctability<.45)]=2
        action[(core_risk==0)&(pflip>=.72)]=3
        action[(core_risk==0)&(pflip<.72)&(readiness<.48)]=1

        scale=np.full(n,1,dtype=np.int64)
        scale[(pflip>=.60)&(local_u+.05<meso_u)]=0
        scale[(longrisk>=.65)|(global_u+.05<meso_u)]=2

        probe=(pflip>=.55).astype(np.float32)
        separability=(sep>=.58).astype(np.float32)

        support_credit=evidence_support*independence
        delta=(.10*support_credit-(.18+.14*np.maximum(severe,irreversibility))*contradiction).astype(np.float32)
        authority_delta=np.stack([delta, .55*delta, .16*np.maximum(delta,0)+.42*np.minimum(delta,0), .08*(support_credit-contradiction)],1).astype(np.float32)

        # Sparse local trust update targets. Event type is softly inferred from conflict/contradiction/support.
        trust_delta=np.zeros((n,3),dtype=np.float32)
        factual_error=(contradiction>.65)&(conflict<.45)
        strategic=(contradiction>.55)&(conflict>=.45)
        factual_truth=(evidence_support>.7)&(~strategic)
        normative_truth=(legitimacy>.7)&(conflict<.3)
        trust_delta[factual_error,0]=-0.20
        trust_delta[factual_error,2]=0.02
        trust_delta[strategic,2]=0.28
        trust_delta[strategic,1]=-0.24
        trust_delta[strategic,0]=-0.00005
        trust_delta[factual_truth,0]=0.06*(1-x[factual_truth,4])
        trust_delta[normative_truth,1]=0.055*(.94-x[normative_truth,5])

        self.x=torch.from_numpy(x)
        self.targets={
            "action":torch.from_numpy(action),
            "scale":torch.from_numpy(scale),
            "probe":torch.from_numpy(probe),
            "separability":torch.from_numpy(separability),
            "authority_delta":torch.from_numpy(authority_delta),
            "trust_delta":torch.from_numpy(trust_delta),
            "core_risk":torch.from_numpy(core_risk),
        }

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], {k:v[idx] for k,v in self.targets.items()}
