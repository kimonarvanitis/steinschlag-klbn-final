# README for Analysis Notebook (analysis.py)

## Overview
This script, titled "Analysis Notebook," is a Jupyter notebook designed for data analysis with a specific focus on comparing distributions and conducting exploratory data analysis (EDA) in two different zones.

## Installation and Requirements
- To run this notebook, certain Python libraries including `matplotlib`, `numpy`, `pandas`, `seaborn`, and `scipy` are required.
- Dependencies can be installed via `pip install -r requirements.txt` and a requirements file can be created using `pip list --format=freeze > requirements.txt`.

## Key Sections of the Notebook
1. **Runtime Configurations**: Sets up the environment and ignores warnings for smoother execution.
2. **Data Import**: Reads data from two zones for analysis.
3. **Explorative Data Analysis (EDA)**: Includes data cleaning, renaming columns for consistency, handling missing values, and summarizing data.
4. **Data Visualization and Statistical Analysis**:
   - Descriptive statistics and visualization using tables, density plots, box plots, histograms, and scatter plots.
   - Statistical distribution fitting for various parameters like speed, mass, and time deltas using libraries like `Fitter` and `scipy.stats`.
   - Generation of Q-Q plots for different distributions to assess the fit.
5. **Comparison of Original and Simulated Data**: Visual comparison of distributions in both zones between original and simulated data.
6. **Simulation Analysis**: Analysis of simulation data, including splitting data by zone and comparing distributions.
7. **Convergence Analysis**: A scatter plot showing the annual probability of fatalities per simulated year.

## Usage
- Execute the notebook cells in order to perform the analysis.
- The script includes several sections for data visualization and statistical analysis which can be modified as needed for different datasets or analysis requirements.

## Special Features
- Detailed exploratory data analysis with visualizations for a clear understanding of the data.
- Comparative analysis of distributions in two different zones.
- Extensive use of statistical distributions and fitting to analyze data.