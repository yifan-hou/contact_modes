import numpy as np

from contact_modes._contact_modes import kernel_basis


def test_null_space():
    np.random.seed(0)
    A = np.random.rand(4,5)
    kern = kernel_basis(A, 1e-8)
    print(A @ kern)
    assert(False)