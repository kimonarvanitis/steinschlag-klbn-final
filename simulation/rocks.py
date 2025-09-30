import numpy.random
import pandas as pd
import scipy.stats
from scipy.stats import *
import numpy as np
from numba import jit

'''
This code snippet is a simulation of rock generation and analysis. It generates rocks with different characteristics in two zones and analyzes if any of the rocks breach a certain condition. The code uses various probability distributions to generate the properties of the rocks and calculates the energy and cumulative mass of the rocks. It then determines if each rock breaches a condition based on its energy, mass, and cumulative mass. The results are stored in a DataFrame and printed for analysis.
'''

# Parameters for the used distribution, see analysis.ipynb for parameter fitting
__distributions = {
    'mass_zone_1': ('gamma', (0.8079614675149851, 11.999999999999998, 822.0119854583972)),
    'speed_zone_1': ('norm', (8.788235294117646, 1.9745088303442118)),
    'time_delta_zone_1': ('expon', (0.0, 30.102941176470587)),
    'speed_zone_2': ('norm', (37.790625, 5.31080027956004)),
    'mass_zone_2': ('gamma', (0.6594803921005904, -3.788074626145447e-27, 132.038825053617)),
    'time_delta_zone_2': ('expon', (0.0, 64.25))
}


def __simulate_distribution(distribution: str, dist_params: tuple, estimated_stones: int):
    dist = globals()[distribution]
    dist.random_state = np.random.RandomState(seed=1)
    # Simulates a distribution of stones based on the specified probability distribution
    return dist.rvs(*dist_params, size=estimated_stones)


def __create_abs_timedelta(time_deltas: list):
    # Converts timedelta since last rock to timedelta since 0
    timestamps = [0.0]
    for delta in time_deltas:
        timestamps.append(np.round(timestamps[-1] + delta))
    return timestamps[1:]


@jit(nopython=True)
def __define_breached_stones_numba(datetimes, masses, energies):
    # Creates new column if a stone breaches the net or not

    # get the amount of stones
    n = len(datetimes)

    breaches = np.zeros(n, dtype=np.int32)
    cum_kg = np.zeros(n, dtype=np.float64)

    cleaning_period = 24 * 3600  # convert hours to seconds for consistency
    hours_passed = 0

    total_mass = 0
    total_energy = 0
    breached = 0

    for i in range(n):

        hours_passed += datetimes[i]
        if hours_passed >= cleaning_period:
            hours_passed = 0
            total_mass = 0
            total_energy = 0
            breached = 0

        cum_kg[i] = total_mass
        total_mass += masses[i]
        total_energy += energies[i]

        if breached == 1 or energies[i] > 1000 or (total_mass > 2000 and energies[i] > 500):
            breached = 1
            breaches[i] = 1

    return breaches, cum_kg


def generate_rocks(estimated_stones):
    stones_z1 = int(estimated_stones * 0.68)
    stones_z2 = estimated_stones - stones_z1

    # Zone 1
    df_z1 = pd.DataFrame({
        'speed_m/s': __simulate_distribution(*__distributions['speed_zone_1'], stones_z1),
        'mass_kg': __simulate_distribution(*__distributions['mass_zone_1'], stones_z1),
        'datetime': __create_abs_timedelta(
            __simulate_distribution(*__distributions['time_delta_zone_1'], stones_z1)),
        'zone': 1,
    })

    # Zone 2
    df_z2 = pd.DataFrame({
        'speed_m/s': __simulate_distribution(*__distributions['speed_zone_2'], stones_z2),
        'mass_kg': __simulate_distribution(*__distributions['mass_zone_2'], stones_z2),
        'datetime': __create_abs_timedelta(
            __simulate_distribution(*__distributions['time_delta_zone_2'], stones_z2)),
        'zone': 2,
    })

    df = pd.concat([df_z1, df_z2], ignore_index=True)
    df['energy'] = 0.5 * df['mass_kg'] * df["speed_m/s"] ** 2 / 1000
    df = df.sort_values(by='datetime')

    # if mass or speed isn't larger than 0, regenrate the mass and speed for the date
    for i, row in df.iterrows():
        while True:

            if row['mass_kg'] <= 0 or row['speed_m/s'] <= 0:
                zone = row['zone']
                if zone == 1:
                    df.at[i, 'mass_kg'] = __simulate_distribution(*__distributions['mass_zone_1'], 1)[0]
                    df.at[i, 'speed_m/s'] = __simulate_distribution(*__distributions['speed_zone_1'], 1)[0]
                elif zone == 2:
                    df.at[i, 'mass_kg'] = __simulate_distribution(*__distributions['mass_zone_2'], 1)[0]
                    df.at[i, 'speed_m/s'] = __simulate_distribution(*__distributions['speed_zone_2'], 1)[0]
                df.at[i, 'energy'] = 0.5 * df.at[i, 'mass_kg'] * df.at[i, "speed_m/s"] ** 2 / 1000

            if df.at[i, 'speed_m/s'] >= 0 and df.at[i, 'mass_kg'] >= 0:
                break

    print('Calculating breaches and cleaning...')
    breaches, cum_kg = __define_breached_stones_numba(df['datetime'].values, df['mass_kg'].values, df['energy'].values)

    df['breaches'] = breaches
    df['cum_kg'] = cum_kg

    df.to_csv('raw_rocks.csv')

    return df['datetime'].max() / 24 / 365, df


if __name__ == '__main__':
    year, df = generate_rocks(10_000)
    print(f"Years: {year}")
    print(df.describe())
    print(df.loc[df['breaches'] == 1, 'breaches'])
