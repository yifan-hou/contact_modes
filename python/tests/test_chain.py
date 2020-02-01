import numpy as np

from contact_modes.shape import Chain

def test_chain():
    chain = Chain(10)
    chain.generate()
    assert(False)