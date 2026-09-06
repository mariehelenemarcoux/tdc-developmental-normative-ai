import torch
from tdc_gen26 import TDCGen26Torch, TDCCompositeLoss
from tdc_gen26.data import SyntheticTDCDataset


def test_torch_forward_shapes():
    m=TDCGen26Torch(input_dim=24,hidden_dim=32)
    x=torch.rand(7,24)
    y=m(x)
    assert y["action_logits"].shape == (7,4)
    assert y["scale_logits"].shape == (7,3)
    assert y["trust_delta"].shape == (7,3)


def test_loss_backward():
    ds=SyntheticTDCDataset(n=16,seed=5)
    x=torch.stack([ds[i][0] for i in range(16)])
    t={k:torch.stack([ds[i][1][k] for i in range(16)]) for k in ds[0][1]}
    m=TDCGen26Torch(input_dim=24,hidden_dim=32)
    loss,_=TDCCompositeLoss()(m(x),t)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in m.parameters())
