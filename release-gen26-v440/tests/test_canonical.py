from tdc_gen26 import CanonicalTDCGen26V440, MultidimensionalTrust


def test_core_protection_precedes_optimization():
    m=CanonicalTDCGen26V440()
    r=m.decide({"irreversibility":.95,"severe_harm":.9,"correctability":.8,"legitimacy":1.0,"role_symmetry":1.0})
    assert r["core_protection_active"] is True
    assert r["action"] in {"TRANSFORM","ABSTAIN"}


def test_meso_default_when_stable():
    m=CanonicalTDCGen26V440()
    r=m.decide({"decision_flip_probability":.1,"local_uncertainty":.4,"meso_uncertainty":.3,"global_uncertainty":.4})
    assert r["scale"] == "MESO18"


def test_directional_review_requires_separability():
    m=CanonicalTDCGen26V440()
    hi=m.decide({"decision_flip_probability":.8,"corrective_harmful_separability":.9})
    lo=m.decide({"decision_flip_probability":.8,"corrective_harmful_separability":.2})
    assert hi["directional_review"] is True
    assert lo["directional_review"] is False


def test_sparse_trust_update_locality():
    t=MultidimensionalTrust(T_E=.8,T_N=.8,T_S=.1)
    before=t.T_E
    t.update("strategic_manipulation")
    assert before-t.T_E < .01
    assert t.T_N < .8 and t.T_S > .1


def test_observation_inference_separation_reported():
    r=CanonicalTDCGen26V440().decide({})
    assert r["diagnostics"]["observation_inference_separated"] is True
