import contact_modes


def test_combinations():
    print(contact_modes.lexographic_combinations(5,2))
    for c in contact_modes.lexographic_combinations(5, 3):
        print(c)
    for c in contact_modes.lexographic_combinations(5, 0):
        print(c)
    for c in contact_modes.lexographic_combinations(5, 5):
        print(c)
    assert(False)