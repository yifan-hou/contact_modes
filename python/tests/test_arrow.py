import numpy as np

from contact_modes.viewer import make_frame


def test_make_frame():
    z = np.random.rand(3,1)
    make_frame(z)
    assert(False)