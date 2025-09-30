import csv
from rocks import generate_rocks
import numpy as np
import pandas as pd

'''
This code snippet imports the necessary libraries (csv, rocks, numpy, pandas) and defines two functions: __generate_unique_times and __get_deadly_accidents.

The __generate_unique_times function generates a list of unique random times for a given number of cars per day.

The __get_deadly_accidents function calculates the number of deadly accidents by comparing a list of times per day with the generated unique times.

The main block of the code initializes a list of years, opens a CSV file, and iterates over the years. For each year, it generates a random number of rocks, filters them based on certain criteria, groups them by day and time, calculates the number of deadly accidents using the __get_deadly_accidents function, and writes the results to the CSV file.
'''

seed = 1

def __generate_unique_times(cars_per_day=1200):
    unique_times = set()
    while len(unique_times) < cars_per_day:
        random_hour = np.random.randint(0, 24)
        random_minute = np.random.randint(0, 60)
        random_second = np.random.randint(0, 60)
        random_time = random_hour + random_minute / 60 + random_second / 3600
        unique_times.add(random_time)
    return list(unique_times)


def __get_deadly_accidents(times_per_day: list) -> int:
    dead = 0
    cars_per_day = 1200
    for times in times_per_day:
        car_times = __generate_unique_times(cars_per_day)
        for time in times:
            for car_time in car_times:
                if time == car_time:
                    dead += 1
                    break
    return dead


if __name__ == '__main__':
    years = []

    for i in range(1, 100):
        years.append(i * 10_000)

    with open("results.csv", "wt") as fp:
        writer = csv.writer(fp, delimiter=",")

        for n_rocks in years:
            # reseed the random number generator
            np.random.seed(seed)

            dur, df = generate_rocks(n_rocks)

            df = df.loc[df['breaches'] == 1]

            # DataFrame nach Datum und Uhrzeit gruppieren
            df_grouped_by_day = df.groupby(df['datetime'] // 24)

            times_per_day = []

            # Iteriere durch die Gruppen und zeige die Zeiten an
            for _, group in df_grouped_by_day:
                tmp = group['datetime'].tolist()
                times_per_day.append([x % 24 for x in tmp])

            deaths = __get_deadly_accidents(times_per_day)

            writer.writerow([n_rocks, deaths, dur, deaths / dur])
            fp.flush()

            print(f"Dead: {deaths}")
            print(f"Years: {dur}")
            print(f"Prob dead per year: {deaths / dur}")
