# Simulation Scripts README

## Overview
This README provides an overview of the `rocks.py` and `simulation.py` scripts, which are designed for simulations in a geological or related context.
Dependencies can be installed via `pip install -r requirements.txt`

## rocks.py

### Purpose:
`rocks.py` simulates the distribution of stones in different zones based on predefined parameters. It also determines whether a stone breaches a net based on its mass, speed, and energy.

### Key Functionalities:
- Defines distributions for mass, speed, and time deltas in two zones.
- Simulates distributions using SciPy's statistical functions.
- Creates a DataFrame for each zone with simulated data on speed, mass, and time.
- Calculates energy and determines if a stone breaches a net.
- Generates a combined DataFrame for both zones and exports it to a CSV file.

### Usage:
Run the script to simulate stone distributions and net breaches for a given number of stones. The output is a detailed DataFrame and a summary of breaches.

## simulation.py

### Purpose:
`simulation.py` extends the functionality of `rocks.py` by simulating potential accidents based on the times when stones breach the net and the times when cars pass by the simulated area.

### Key Functionalities:
- Generates unique times representing when cars pass by an area.
- Determines the number of deadly accidents based on the overlap of times between car passages and stone breaches.
- Runs simulations for different numbers of stones and calculates the probability of accidents per year.

### Usage:
Execute the script to perform a series of simulations over varying years, calculating the number of deadly accidents. The results are saved to a CSV file.