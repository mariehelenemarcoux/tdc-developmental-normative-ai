from tdc_gen26.data import SyntheticTDCDataset, FEATURE_NAMES


def test_dataset_contract():
    ds=SyntheticTDCDataset(n=32,seed=123)
    assert len(FEATURE_NAMES)==24
    assert ds[0][0].shape[0]==24
    assert set(ds[0][1])=={"action","scale","probe","separability","authority_delta","trust_delta","core_risk"}
