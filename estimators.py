from scipy.stats import weibull_min
from scipy import stats
import numpy as np



def estimate_all(numbers_lists, distributions):
    result = []
    for i in range(0, len(numbers_lists)):
        distribution = distributions[i]
        numbers = numbers_lists[i]
        if distribution == "exp":
            result.append(estimate_exponential(numbers))
        elif distribution == "weibull":
            result.append(estimate_weibull(numbers))
        else:
            result.append(estimate_lognormal(numbers))
    return result

def estimate_exponential(numbers):
    lambd = len(numbers) / sum(numbers)
    return lambd

def estimate_weibull(numbers):
    shape, loc, scale = weibull_min.fit(numbers, floc=0)
    return shape, scale

def estimate_lognormal(numbers):
    sigma, loc, scale = stats.lognorm.fit(numbers, floc=0)
    # get mu
    mu = np.log(scale)
    return mu, sigma
