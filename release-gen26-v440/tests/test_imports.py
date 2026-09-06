def test_imports():
    import tdc_gen26
    from tdc_gen26 import CanonicalTDCGen26V440, TDCGen26Torch, TDCCompositeLoss
    assert tdc_gen26.REFERENCE_VERSION == "v440"
    assert CanonicalTDCGen26V440.reference_version == "v440"
    assert TDCGen26Torch is not None
    assert TDCCompositeLoss is not None
