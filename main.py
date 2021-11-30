# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random

import scipy
from scipy.special import erfinv, gamma, factorial
import estimators
import matplotlib.pyplot as plt
import chaospy
import numpy as np
from scipy.stats import t, sem, weibull_min
import statistics
import subSystems
import pandas as pd




#### A


def get_500_random_numbers_uniform(seed, amount):
    random.seed(seed)
    res = []
    for i in range(0, amount):
        res.append(random.uniform(0, 1))

    return res


##### B 1

def convert_to_exponential(lambd, uniform_numbers):
    res = []
    for number in uniform_numbers:
        # the opposite function of exponential
        exponential_number = (-1 / lambd) * np.log(1 - number)  # log is lan
        res.append(exponential_number)

    return res


def convert_to_weibull(alpha, beta, uniform_numbers):
    res = []
    for number in uniform_numbers:
        # the opposite function of weibull
        # weibull_number = ((-1 / alpha) * np.log(1 - number)) ** (1 / beta)  # log is lan
        weibull_number = alpha * ((-np.log(1 - number)) ** (1 / beta))  # log is lan
        res.append(weibull_number)

    return res


def convert_to_lognormal(mu, sigma, uniform_numbers):
    res = []
    for number in uniform_numbers:
        # the opposite function of lognormal
        lognormal_number = math.e ** (mu + math.sqrt(2 * (sigma) ** 2) * erfinv(2 * number - 1))
        res.append(lognormal_number)

    return res


def generate_numbers(random_numbers):
    res_numbers = []

    # 1
    MC = convert_to_exponential(lambd=(1 / 12000), uniform_numbers=random_numbers)

    res_numbers.append(MC)

    # 2
    MMR = convert_to_exponential(lambd=(1 / 26000), uniform_numbers=random_numbers)
    res_numbers.append(MMR)

    # 3
    GPS_ANT = convert_to_weibull(beta=0.98, alpha=26213, uniform_numbers=random_numbers)
    res_numbers.append(GPS_ANT)

    # 4
    LOC_ANT_Swi = convert_to_lognormal(mu=9.86, sigma=1.31, uniform_numbers=random_numbers)
    res_numbers.append(LOC_ANT_Swi)

    # 5
    GS_ANT = convert_to_weibull(beta=1.01, alpha=25326, uniform_numbers=random_numbers)
    res_numbers.append(GS_ANT)

    # 6
    LOC_ANT = convert_to_weibull(beta=0.86, alpha=31636, uniform_numbers=random_numbers)
    res_numbers.append(LOC_ANT)

    # 7
    RA = convert_to_exponential(lambd=(1 / 80000), uniform_numbers=random_numbers)
    res_numbers.append(RA)

    # 8
    RA_ANT = convert_to_weibull(beta=1.23, alpha=35380, uniform_numbers=random_numbers)
    res_numbers.append(RA_ANT)

    # 9
    NAV_4000 = convert_to_exponential(lambd=(1 / 20000), uniform_numbers=random_numbers)
    res_numbers.append(NAV_4000)

    # 10
    VOR_ANT = convert_to_weibull(beta=1.15, alpha=28263, uniform_numbers=random_numbers)
    res_numbers.append(VOR_ANT)

    # 11
    MB_ANT = convert_to_weibull(beta=0.92, alpha=24926, uniform_numbers=random_numbers)
    res_numbers.append(MB_ANT)

    # 12
    ADF_ANT = convert_to_weibull(beta=0.99, alpha=21042, uniform_numbers=random_numbers)
    res_numbers.append(ADF_ANT)

    # 13
    DME_INT = convert_to_exponential(lambd=(1 / 50000), uniform_numbers=random_numbers)
    res_numbers.append(DME_INT)

    # 14
    ANT_42 = convert_to_weibull(beta=0.88, alpha=51656, uniform_numbers=random_numbers)
    res_numbers.append(ANT_42)

    # write_to_excel.write_res_to_excel(res_names, res_numbers)

    return res_numbers


def build_histogram(numbers):
    plt.hist(numbers, bins=10)
    plt.show()


def q_1_c(distributions_amount, distributions):
    # initialization
    seed = 1  # changes every iteration
    all_estimators = []
    for i in range(0, distributions_amount):
        if distributions[i] == "exp":
            # exp distribution has only one param - needs one list
            all_estimators.append([])
        else:
            # different list to each param
            all_estimators.append([[], []])

    for i in range(0, 100):
        random_numbers = get_500_random_numbers_uniform(seed=seed, amount=500)
        numbers_lists = generate_numbers(random_numbers)
        cur_estemators = estimators.estimate_all(numbers_lists, distributions)

        for j in range(0, distributions_amount):
            if distributions[j] == "exp":
                all_estimators[j].append(cur_estemators[j])
            else:
                # two lists for two params
                all_estimators[j][0].append(cur_estemators[j][0])
                all_estimators[j][1].append(cur_estemators[j][1])

        seed += 1

    # now all_estimators is ready
    return calculate_all_confidence_interval(all_estimators, distributions)


def calculate_all_confidence_interval(all_estimators, distributions):
    # revah bae semah

    confidence_intervals_normal = []
    confidence_intervals_empirical = []

    for i in range(0, len(all_estimators)):
        if distributions[i] == "exp":  # has 1 param
            # estimators
            cur_estimators = all_estimators[i]

            # normal
            cur_confidence_interval_normal = calculate_single_confidence_interval_normal(cur_estimators)
            confidence_intervals_normal.append(cur_confidence_interval_normal)

            # empirical
            cur_confidence_interval_empirical = calculate_single_confidence_interval_empirical(cur_estimators)
            confidence_intervals_empirical.append(cur_confidence_interval_empirical)
        else:  # has 2 params
            # estimators
            cur_estimators_param_1 = all_estimators[i][0]
            cur_estimators_param_2 = all_estimators[i][1]

            # normal
            cur_confidence_interval_normal_1 = calculate_single_confidence_interval_normal(cur_estimators_param_1)
            cur_confidence_interval_normal_2 = calculate_single_confidence_interval_normal(cur_estimators_param_2)

            confidence_intervals_normal.append((cur_confidence_interval_normal_1, cur_confidence_interval_normal_2))

            # empirical
            cur_confidence_interval_empirical_1 = calculate_single_confidence_interval_empirical(cur_estimators_param_1)
            cur_confidence_interval_empirical_2 = calculate_single_confidence_interval_empirical(cur_estimators_param_2)

            confidence_intervals_empirical.append((cur_confidence_interval_empirical_1, cur_confidence_interval_empirical_2))

    return confidence_intervals_normal, confidence_intervals_empirical


def calculate_single_confidence_interval_normal(numbers):
    numbers = np.array(numbers)

    m = numbers.mean()
    s = numbers.std()
    dof = len(numbers) - 1
    confidence = 0.9

    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))

    return (m - s * t_crit / np.sqrt(len(numbers)), m + s * t_crit / np.sqrt(len(numbers)))


def calculate_single_confidence_interval_empirical(numbers):
    sorted_numbers = sorted(numbers)
    low_section = sorted_numbers[:int(0.05 * len(numbers))]
    high_section = sorted_numbers[int(0.95 * len(numbers)):]
    cur_interval = low_section + high_section
    cur_interval = np.array(cur_interval)

    return t.interval(0.9, len(cur_interval), cur_interval.mean(), sem(cur_interval))


def get_random_numbers_halton(amount):
    uniform = chaospy.Uniform(0, 1)
    return list(uniform.sample(amount, rule='halton'))

def q_1_d(distributions):
    # halton function
    sizes = [50, 200, 500]
    res = []

    for cur_size in sizes:
        initial_random_numbers_halton = get_random_numbers_halton(cur_size)
        distributions_numbers_halton = generate_numbers(initial_random_numbers_halton)
        estemated_halton = estimators.estimate_all(distributions_numbers_halton, distributions)
        res.append(estemated_halton)

    return res


def q_2_a():
    MC = subSystems.random_numbers_MC()
    #MC = generate_numbers(get_500_random_numbers_uniform(0.5, 500))[0]
    MMR = subSystems.random_numbers_MMR()
    RA = subSystems.random_numbers_RA()
    VHF_NAV = subSystems.random_numbers_VHF_NAV()
    DME = subSystems.random_numbers_DME()
    RNS = subSystems.random_numbers_RNS([MC, MMR, RA, VHF_NAV, DME])
    all_sub_system = [MC, MMR, RA, VHF_NAV, DME, RNS]
    for lst in all_sub_system:
        print(q_2_a_helper(lst))


def q_2_a_helper(random_nums):
    # build_histogram(random_nums)
    median = statistics.median(random_nums)
    data = np.array(random_nums)
    mean = data.mean()
    top = np.percentile(data, 90)
    buttom = np.percentile(data, 10)

    return mean, median, top, buttom

def q2_b():
    #MC = generate_numbers(get_500_random_numbers_uniform(0.5, 500))[0]
    MC = subSystems.random_numbers_MC()
    MMR = subSystems.random_numbers_MMR()
    RA = subSystems.random_numbers_RA()
    VHF_NAV = subSystems.random_numbers_VHF_NAV()
    DME = subSystems.random_numbers_DME()
    RNS = subSystems.random_numbers_RNS([MC, MMR, RA, VHF_NAV, DME])
    all_sub_system = [MC, MMR, RA, VHF_NAV, DME, RNS]
    all_probability = [0,0,0,0,0,0]
    indx = 0
    for lst in all_sub_system:
        for num in lst:
            if num > 1400:
                all_probability[indx] += 1
        indx += 1
    for i in range(6):
        all_probability[i] = all_probability[i]/500

    return all_probability




if __name__ == '__main__':
    # data

    exp = "exp"
    weibull = "weibull"
    lognormal = "lognormal"
    distributions = [exp, exp, weibull, lognormal, weibull, weibull, exp, weibull, exp, weibull, weibull, weibull, exp,
                     weibull]
    res_names = ['MC', 'MMR', 'GPS_ANT', 'LOC_ANT_Swi', 'GS_ANT', 'LOC_ANT', 'RA', 'RA_ANT', 'NAV_4000', 'VOR_ANT',
                 'MB_ANT', 'ADF_ANT', 'DME_INT', 'ANT_42']
    distributions_amount = len(res_names)

    # 1
    # random_numbers = get_500_random_numbers_uniform(seed=0.5, amount=500)
    #
    # res_numbers1 = generate_numbers(random_numbers)
    #
    # # 1 A
    #
    # # 1 A 1
    # # build_histogram(res_numbers1[13])  # 14
    #
    # # 1 A 2
    # # need to hold the result or write it down !
    # all_estemated = estimators.estimate_all(res_numbers1, distributions)
    #
    # # 1 B
    #
    # # 1 B 1 - nothing to do
    # # 1 B 2 - nothing to do
    # # 1 B 3
    # random_numbers = get_500_random_numbers_uniform(seed=0.7, amount=10000)
    # res_numbers_1_B_3 = generate_numbers(random_numbers)
    # estemators_1_B_3 = estimators.estimate_all(res_numbers_1_B_3, distributions)

    # 1 C

    # 1 C 1

    # confidence_intervals_q_1_c = q_1_c(distributions_amount, distributions)

    # 1 D

    # relevant_estimators_halton = q_1_d(distributions)

    # 2 A
    print(q_2_a())
    print("--------------")
    print(q2_b())
    end = "end"
