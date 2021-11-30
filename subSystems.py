import numpy as np
from scipy.stats import t, sem, weibull_min

def random_numbers_MC():
    random_nums = []
    for i in range(500):
        mc = np.random.exponential(scale=12000, size=1)[0]
        random_nums.append(mc)
    return random_nums

def random_numbers_MMR():
    random_nums = []
    for i in range(0,500):
        MMR_1 = np.random.exponential(scale=26000, size=1)[0]
        GPS_ANT_1 = (weibull_min.rvs(0.98, size=1) * 26213)[0]
        Loc_ANT_1 = np.random.lognormal(9.86, 1.31, size=1)[0]
        total_1 = min(MMR_1, GPS_ANT_1, Loc_ANT_1)
        MMR_2 = np.random.exponential(scale=26000, size=1)[0]
        GPS_ANT_2 = (weibull_min.rvs(0.98, size=1) * 26213)[0]
        Loc_ANT_2 = np.random.lognormal(9.86, 1.31, size=1)[0]
        total_2 = min(MMR_2, GPS_ANT_2, Loc_ANT_2)
        GS_ANT = (weibull_min.rvs(1.01, size=1) * 25326)[0]
        LOC_ANT = (weibull_min.rvs(0.86, size=1) * 31636)[0]

        result = min(max(total_1, total_2), GS_ANT, LOC_ANT)
        random_nums.append(result)
    return random_nums


def random_numbers_RA():
    random_nums = []
    for i in range(0, 500):
        RA_1 = np.random.exponential(scale=80000, size=1)[0]
        RA_ANT_1 = (weibull_min.rvs(1.23, size=1) * 35380)[0]
        RA_ANT_2 = (weibull_min.rvs(1.23, size=1) * 35380)[0]
        total_1 = min(RA_1, RA_ANT_1, RA_ANT_2)
        RA_2 = np.random.exponential(scale=80000, size=1)[0]
        RA_ANT_2 = (weibull_min.rvs(1.23, size=1) * 35380)[0]
        RA_ANT_3 = (weibull_min.rvs(1.23, size=1) * 35380)[0]
        total_2 = min(RA_2, RA_ANT_2, RA_ANT_3)


        result = max(total_1, total_2)
        random_nums.append(result)
    return random_nums


def random_numbers_VHF_NAV():
    random_nums = []
    for i in range(0, 500):
        NAV4000_1 = np.random.exponential(scale=20000, size=1)[0]
        NAV4000_2 = np.random.exponential(scale=20000, size=1)[0]
        total_1 = max(NAV4000_1, NAV4000_2)
        VOR_ANT =(weibull_min.rvs(1.15, size=1) * 28263)[0]
        MB_ANT =(weibull_min.rvs(0.92, size=1) * 24926)[0]
        ADF_ANT =(weibull_min.rvs(0.99, size=1) * 21042)[0]

        result = min(total_1, VOR_ANT, MB_ANT, ADF_ANT)
        random_nums.append(result)
    return random_nums


def random_numbers_DME():
    random_nums = []
    for i in range(0, 500):
        DME_INT_1 = np.random.exponential(scale=50000, size=1)[0]
        ANT42_1 = (weibull_min.rvs(0.88, size=1) * 51656)[0]
        total_1 = min(DME_INT_1, ANT42_1)

        DME_INT_2 = np.random.exponential(scale=50000, size=1)[0]
        ANT42_2 = (weibull_min.rvs(0.88, size=1) * 51656)[0]
        total_2 = min(DME_INT_2, ANT42_2)

        result = max(total_1, total_2)
        random_nums.append(result)
    return random_nums


def random_numbers_RNS(all_sub_systems):
    random_nums = all_sub_systems[0].copy()
    for col in range(0,len(all_sub_systems[0])):
        for row in range(1, len(all_sub_systems)):
            if all_sub_systems[row][col] < random_nums[col]:
                random_nums[col] = all_sub_systems[row][col]
    return random_nums

