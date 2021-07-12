import numpy as np


def features_extract_func(instance):
    return [instance.cpu, instance.memory, instance.disk]


def features_normalize_func(x):
    y = np.array(x)
    return y