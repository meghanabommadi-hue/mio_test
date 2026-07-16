import pytest


@pytest.fixture(scope="session")
def synth():
    from miotts import IndicMioSynthesizer

    s = IndicMioSynthesizer()
    s.load()
    return s
