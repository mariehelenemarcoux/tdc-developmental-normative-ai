from __future__ import annotations
import argparse, json
from .runner import train_from_config, evaluate_from_config


def train_main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.json"); p.add_argument("--checkpoint",default=None)
    a=p.parse_args(); print(json.dumps(train_from_config(a.config,a.checkpoint),indent=2))


def evaluate_main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.json"); p.add_argument("--checkpoint",required=True)
    a=p.parse_args(); print(json.dumps(evaluate_from_config(a.config,a.checkpoint),indent=2))
