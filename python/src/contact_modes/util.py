import sys, os
import git
import numpy as np


def get_root():
    path = os.path.realpath(__file__)
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return os.path.join(git_root, 'python')

def get_data():
    return os.path.join(get_root(), 'data')

def get_color(name):
    if name.lower() == 'red':
        return np.array([1, 0.017, 0.017], dtype='float32')
    if name.lower() == 'blue':
        return np.array([0, 0.2588, 1.0], dtype='float32')
    if name.lower() == 'teal':
        return np.array([0.098, 0.902, 0.7255], dtype='float32')
    if name.lower() == 'purple':
        return np.array([0.3294, 0.0, 0.5059], dtype='float32')
    if name.lower() == 'yellow':
        return np.array([1, 0.9882, 0.0040], dtype='float32')
    if name.lower() == 'orange':
        return np.array([0.9961, 0.7294, 0.0549], dtype='float32')
    if name.lower() == 'green':
        return np.array([0.1255, 0.7530, 0.0], dtype='float32')
    if name.lower() == 'pink':
        return np.array([0.898, 0.3569, 0.6902], dtype='float32')
    if name.lower() == 'light blue':
        return np.array([0.4941, 0.7490, 0.9451], dtype='float32')
    if name.lower() == 'dark green':
        return np.array([0.0627, 0.3843, 0.2745], dtype='float32')
    if name.lower() == 'brown':
        return np.array([0.3059, 0.1647, 0.0157], dtype='float32')
    if name.lower() == 'clay':
        return np.array([1.0, 0.5, 0.31], dtype='float32')
    if name.lower() == 'black':
        return np.array([0.0, 0.0, 0.0], dtype='float32')
    if name.lower() == 'white':
        return np.array([1.0, 1.0, 1.0], dtype='float32')
    if name.lower() == 'gray':
        # return np.array([83/255, 98/255, 103/255.], dtype='float32') # gunmetal
        return np.array([102/255, 153/255, 204/255.], dtype='float32') # blue-gray
    print('Unrecognized color:', name)
    assert(False)