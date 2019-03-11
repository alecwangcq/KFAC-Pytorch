from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    else:
        raise NotImplementedError