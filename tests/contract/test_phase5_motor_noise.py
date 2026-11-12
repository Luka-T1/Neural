"""Ugovor: referentni dt motornog šuma i njegov izvor u PARAMS_META ostaju kako su dokumentovani."""

from neural.body import motor


def test_reference_dt_and_noise_source_documented():
    assert motor.REF_DT_MS == 0.1
    src = motor.PARAMS_META["noise"]["source"]
    assert "ASSUMPTION" in src and "REF_DT_MS" in src and "sqrt" in src


def test_noise_scale_is_local_to_body():
    import inspect

    assert "neural.organism" not in inspect.getsource(motor)
    assert motor.noise_scale(motor.REF_DT_MS) == 1.0
