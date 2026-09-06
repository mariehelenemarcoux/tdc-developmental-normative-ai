from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .configs import load_config
from .data import SyntheticTDCDataset
from .losses import TDCCompositeLoss
from .pytorch_model import TDCGen26Torch


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    requested = cfg.get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_from_config(config_path: str, checkpoint_path: str | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    torch.manual_seed(int(cfg["seed"]))
    device = _device_from_config(cfg)
    ds = SyntheticTDCDataset(n=int(cfg["train_samples"]), seed=int(cfg["seed"]), input_dim=int(cfg["input_dim"]))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(int(cfg["seed"])))
    model = TDCGen26Torch(input_dim=int(cfg["input_dim"]), hidden_dim=int(cfg["hidden_dim"])).to(device)
    loss_fn = TDCCompositeLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))

    history=[]
    model.train()
    for epoch in range(int(cfg["epochs"])):
        total=0.0; count=0
        for x,t in loader:
            x=x.to(device); t={k:v.to(device) for k,v in t.items()}
            opt.zero_grad(set_to_none=True)
            out=model(x)
            loss,_=loss_fn(out,t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach())*len(x); count += len(x)
        history.append({"epoch":epoch+1,"loss":total/max(count,1)})

    if checkpoint_path is None:
        checkpoint_path=cfg.get("checkpoint_path","checkpoints/smoke_model.pt")
    cp=Path(checkpoint_path); cp.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":model.state_dict(),
        "config":cfg,
        "reference_version":"v440",
        "implementation_class":"PYTORCH_SCAFFOLD",
        "history":history,
    },cp)
    return {"checkpoint":str(cp),"device":str(device),"history":history,"final_loss":history[-1]["loss"]}


def evaluate_from_config(config_path: str, checkpoint_path: str) -> Dict[str, Any]:
    cfg=load_config(config_path)
    device=_device_from_config(cfg)
    payload=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model=TDCGen26Torch(input_dim=int(cfg["input_dim"]),hidden_dim=int(cfg["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    ds=SyntheticTDCDataset(n=int(cfg["eval_samples"]),seed=int(cfg["eval_seed"]),input_dim=int(cfg["input_dim"]))
    loader=DataLoader(ds,batch_size=int(cfg["batch_size"]),shuffle=False)
    loss_fn=TDCCompositeLoss()
    model.eval()
    total=0.0; n=0; action_ok=0; scale_ok=0; probe_ok=0; sep_ok=0
    with torch.no_grad():
        for x,t in loader:
            x=x.to(device); t={k:v.to(device) for k,v in t.items()}
            out=model(x); loss,_=loss_fn(out,t)
            bs=len(x); total+=float(loss)*bs; n+=bs
            action_ok += int((out["action_logits"].argmax(-1)==t["action"]).sum())
            scale_ok += int((out["scale_logits"].argmax(-1)==t["scale"]).sum())
            probe_ok += int(((torch.sigmoid(out["probe_logit"])>=.5)==(t["probe"]>=.5)).sum())
            sep_ok += int(((torch.sigmoid(out["separability_logit"])>=.5)==(t["separability"]>=.5)).sum())
    return {
        "device":str(device),"samples":n,"loss":total/max(n,1),
        "action_accuracy":action_ok/max(n,1),"scale_accuracy":scale_ok/max(n,1),
        "probe_accuracy":probe_ok/max(n,1),"separability_accuracy":sep_ok/max(n,1),
        "reference_version":"v440",
    }
