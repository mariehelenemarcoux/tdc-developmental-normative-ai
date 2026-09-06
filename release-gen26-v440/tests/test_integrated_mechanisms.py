from tdc_gen26 import ThirdFactorArbiter, RecursiveZoomController, TemporalRegimeState, budgeted_voi


def test_third_factor_core_block():
    r=ThirdFactorArbiter().review(.9,.9,.1,True,.9)
    assert r.authorization == "REJECT_OR_TRANSFORM"


def test_recursive_zoom_earned_by_fragility_and_separability():
    z=RecursiveZoomController(max_depth=4,unit_cost=.08)
    hi=z.plan(.9,.9,.2,.40)
    low=z.plan(.9,.05,.2,.40)
    assert hi.depth > low.depth


def test_temporal_regime_change_detection():
    r=TemporalRegimeState()
    for _ in range(20): r.update(0.0)
    before=r.change_probability
    for _ in range(4): r.update(2.0)
    assert r.change_probability > before


def test_budgeted_voi_cost_discipline():
    assert budgeted_voi(.2,.05,.1).probe is True
    assert budgeted_voi(.2,.15,.1).probe is False
