# %% [markdown]
# # Analysis Notebook 

# %%
# install pip packages 
#! pip install -r requirements.txt
# create requirements file
#!pip list --format=freeze > requirements.txt

# %%
# used libs 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from fitter import Fitter, get_common_distributions
from scipy.stats import dweibull
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import geninvgauss
from scipy.stats import norm
from scipy.stats import norminvgauss
from scipy.stats import probplot
from scipy.stats import skewnorm
from scipy.stats import truncweibull_min
from scipy.stats import genhyperbolic
from scipy.stats import truncweibull_min
from scipy.stats import argus

# %% [markdown]
# ## 1. runtime configurations

# %%
# ignore warnings
warnings.filterwarnings('ignore')
# param dict for fitting distributions
param_dict = {}

# %% [markdown]
# ## 2. import Data 

# %%
# read data zone 1 and 2
zone_1 = pd.read_csv('../data/zone_1.csv', usecols=[0, 1, 2, 3])
zone_2 = pd.read_csv('../data/zone_2.csv', usecols=[0, 1, 2, 3])

# %% [markdown]
# ## 3. explorative data analysis

# %%
# make a copy of the original df with specific rows keeps only the relevant ones
zone_1_cleaned = zone_1[['Datum', 'Uhrzeit', 'Masse [kg]', 'Geschwindigkeit [m/s]']].copy()
zone_2_cleaned = zone_2[['Date', 'Uhrzeit', 'm [kg]', 'v [m/s]']].copy()
# apply a consistent naming 
zone_1_cleaned.columns = ['Date', 'Time', 'Mass_kg', 'Speed_m/s']
zone_2_cleaned.columns = ['Date', 'Time', 'Mass_kg', 'Speed_m/s']

# %%
# sum of is na zone 1
zone_1_cleaned.isna().sum()

# %%
# sum of is na zone 2
zone_2_cleaned.isna().sum()

# %%
zone_2_cleaned[0:10]

# %%
# drop all duplicates and missing values of zone 1 and 2
# zone 1 
zone_1_cleaned.drop_duplicates(inplace=True)
zone_1_cleaned.dropna(inplace=True)
# zone 2
zone_2_cleaned.drop_duplicates(inplace=True)
zone_2_cleaned.dropna(inplace=True)

# %%
zone_1_cleaned.describe()

# %%
zone_2_cleaned.describe()

# %%
# calculate the ratio of both zones for simulations 
ratio = zone_1_cleaned.count() / zone_2_cleaned.count()
ratio

# %%
# combine Date and Time columns to one and format it to a datetime-Object
# zone 1
zone_1_cleaned['DateTime'] = pd.to_datetime(zone_1_cleaned['Date'] + ' ' + zone_1_cleaned['Time'])
# zone 2
zone_2_cleaned['DateTime'] = pd.to_datetime(zone_2_cleaned['Date'] + ' ' + zone_2_cleaned['Time'])

# %%
# drop the old Date and Time columns
# zone 1
zone_1_cleaned.drop(['Date', 'Time'], axis=1, inplace=True)
# zone 2
zone_2_cleaned.drop(['Date', 'Time'], axis=1, inplace=True)

# %%
# zone 1
# calculate time deltas
# Calculating time deltas in hours
# position 0 is NAN because there is no row before the first row
zone_1_cleaned['Time_Delta_hours'] = zone_1_cleaned['DateTime'].diff().dt.total_seconds() / 3600  # convert to hours

# replace the NAN Value with 0.0
zone_1_cleaned['Time_Delta_hours'].iloc[0] = 0.0

# add energy in kj
zone_1_cleaned['kj'] = 0.5 * zone_1_cleaned['Mass_kg'] * zone_1_cleaned['Speed_m/s'] ** 2 / 1000

# zone 2
# calculate time deltas
# Calculating time deltas in hours
# position 0 is NAN because there is no row before the first row
zone_2_cleaned['Time_Delta_hours'] = zone_2_cleaned['DateTime'].diff().dt.total_seconds() / 3600  # convert to hours

# replace the NAN Value with 0.0
zone_2_cleaned['Time_Delta_hours'].iloc[0] = 0.0

# add energy in kj
zone_2_cleaned['kj'] = 0.5 * zone_2_cleaned['Mass_kg'] * zone_2_cleaned['Speed_m/s'] ** 2 / 1000

# %%
# zone 1
zone_1_cleaned.head(10)

# %%
# zone 1
zone_2_cleaned.head(10)

# %%
#Auswahl der relevanten Variablen für die Tabelle
relevant_var = ['Mass_kg', 'Speed_m/s', 'Time_Delta_hours']  # Ersetzen Sie dies durch Ihre Spaltennamen
zone_1_cleaned_stats = zone_1_cleaned[relevant_var]

# Berechnung der deskriptiven Statistiken und Runden
descriptive_stats1 = zone_1_cleaned_stats.describe().round(2)

# Setzt die Skalierung der Schriftgröße für die Grafik
sns.set(font_scale=1.2)

# Erstellt eine Figur und ein einzelnes Unterdiagramm
fig, ax = plt.subplots(figsize=(10, 6))

# Die Achsen werden unsichtbar gemacht
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Der Rahmen des Unterdiagramms wird ausgeblendet
ax.set_frame_on(False)

# Erstellt die Tabelle mit der 'table' Funktion von Matplotlib
tab = plt.table(cellText=descriptive_stats1.values, colLabels=descriptive_stats1.columns, 
                rowLabels=descriptive_stats1.index, cellLoc='center', rowLoc='center', 
                loc='center')

# Verbesserung der Ästhetik der Tabelle
tab.auto_set_font_size(False)  # Automatische Schriftgrößeneinstellung wird deaktiviert
tab.set_fontsize(12)           # Schriftgröße für die Tabelle setzen
tab.scale(1.2, 1.2)            # Skalierung der Tabelle anpassen

# Titel für die Grafik setzen
plt.title('Deskriptive Statistik - Zone 1')

# Zeigt die Grafik an
plt.show()

# %%
#Auswahl der relevanten Variablen für die Tabelle
relevant_var = ['Mass_kg', 'Speed_m/s', 'Time_Delta_hours']  # Ersetzen Sie dies durch Ihre Spaltennamen
zone_2_cleaned_stats = zone_2_cleaned[relevant_var]

# Berechnung der deskriptiven Statistiken und Runden
descriptive_stats2 = zone_2_cleaned_stats.describe().round(2)

# Setzt die Skalierung der Schriftgröße für die Grafik
sns.set(font_scale=1.2)

# Erstellt eine Figur und ein einzelnes Unterdiagramm
fig, ax = plt.subplots(figsize=(10, 6))

# Die Achsen werden unsichtbar gemacht
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Der Rahmen des Unterdiagramms wird ausgeblendet
ax.set_frame_on(False)

# Erstellt die Tabelle mit der 'table' Funktion von Matplotlib
tab = plt.table(cellText=descriptive_stats2.values, colLabels=descriptive_stats2.columns, 
                rowLabels=descriptive_stats2.index, cellLoc='center', rowLoc='center', 
                loc='center')

# Verbesserung der Ästhetik der Tabelle
tab.auto_set_font_size(False)  # Automatische Schriftgrößeneinstellung wird deaktiviert
tab.set_fontsize(12)           # Schriftgröße für die Tabelle setzen
tab.scale(1.2, 1.2)            # Skalierung der Tabelle anpassen

# Titel für die Grafik setzen
plt.title('Deskriptive Statistik - Zone 2')

# Zeigt die Grafik an
plt.show()

# %% [markdown]
# ## 3.1 zones comparing

# %%
# zone 1 & 2 compare Mass and Speed
fig, ax = plt.subplots()
ax.scatter(zone_1_cleaned['Mass_kg'], zone_1_cleaned['Speed_m/s'], c='blue', label='Zone 1')
ax.scatter(zone_2_cleaned['Mass_kg'], zone_2_cleaned['Speed_m/s'], c='red', label='Zone 2')
ax.legend()
ax.axes.set_xlabel('Masse [kg]')
ax.axes.set_ylabel('Geschwindigkeit [m/s]')
n = zone_1_cleaned.shape[0] + zone_2_cleaned.shape[0]
plt.title(f'Scatterplot: Masse und Geschwindigkeit\nAnzahl der gemessenen Punkte: {n}')
plt.show()

# %% [markdown]
# ## 3.2 zone 1 

# %% [markdown]
# ### 3.2.1 zone 1 speed

# %%
# Setzen Sie den Hintergrund auf weiß
sns.set(style="whitegrid")
# density plot of zone 1 speeds
sns.distplot(zone_1_cleaned['Speed_m/s'])
# add title 
plt.title('Dichtekurve der Geschwindigkeiten in Zone 1')
#X-Achse beschriften
plt.xlabel('Geschwindigkeit [m/s]')

# %%
# sns boxplot for zone 1
sns.boxplot(data=zone_1_cleaned, x='Speed_m/s')
# title of the plot
plt.title('Boxplot of zone 1 speed')

# %%
# Setzen Sie den Hintergrund auf weiß mit Gitterlinien
sns.set(style="whitegrid")

# Erstellen Sie eine Figur mit zwei Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Histogramm der Geschwindigkeiten in Zone 1
sns.histplot(zone_1_cleaned['Speed_m/s'], kde=False, ax=ax1)
ax1.set_title('Histogramm der Geschwindigkeiten in Zone 1')
ax1.set_xlabel('Geschwindigkeit [m/s]')

# Boxplot der Geschwindigkeiten in Zone 1
sns.boxplot(data=zone_1_cleaned, x='Speed_m/s', ax=ax2)
ax2.set_title('Boxplot Geschwindigkeiten in Zone 1')
ax2.set_xlabel('Geschwindigkeit [m/s]')

# Haupttitel hinzufügen
plt.suptitle("Visuelle Datenanalyse der Geschwindigkeit", fontsize=16)


# Layout anpassen, um Überlappungen zu verhindern
plt.tight_layout()

# Zeigen Sie die kombinierten Plots an
plt.show()

# %%
# fitter to get an idea how distributions look like
# original fit
#f = Fitter(zone_1_cleaned['Speed_m/s'])
f = Fitter(zone_1_cleaned['Speed_m/s'], distributions=get_common_distributions())
f.fit()
f.summary()

# %% [markdown]
# different QQ-Plot for different distributions

# %%
# Fit gamma distribution to speed
shape, loc, scale = gamma.fit(zone_1_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Speed_m/s']), dist=gamma, sparams=(shape, loc, scale), plot=plt)
# Adding title to plot
plt.title('QQ-Plot Gamma-Verteilung Speed km/h')
# Showing plot
plt.show()

# %%
# Fit skewnorm distribution to data
shape, loc, scale = skewnorm.fit(zone_1_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Speed_m/s']), dist=skewnorm, sparams=(shape, loc, scale), plot=plt)
# Adding title to plot
plt.title("QQ-Plot skewnorm-Verteilung Speed km/h")
# Showing plot
plt.show()

# %%
# Fit norm distribution to data
dist_params = norm.fit(zone_1_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Speed_m/s']), dist=norm, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot norm-Verteilung Speed km/h")
# Showing plot
plt.show()

# %%
#Zeige die QQ-Plots von den Verteilungen der Geschwindigkeit

# Vorbereitung der Grafik mit zwei Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Hintergrund der Grafik
fig.patch.set_facecolor('white')

# QQ-Plot für die Normalverteilung
dist_params = norm.fit(zone_1_cleaned['Speed_m/s'])
probplot(np.array(zone_1_cleaned['Speed_m/s']), dist=norm, sparams=dist_params, plot=ax1)
ax1.set_title("QQ-Plot Normalverteilung Geschwindigkeit [m/s]")

# QQ-Plot für die Gammaverteilung
shape, loc, scale = gamma.fit(zone_1_cleaned['Speed_m/s'])
probplot(np.array(zone_1_cleaned['Speed_m/s']), dist=gamma, sparams=(shape, loc, scale), plot=ax2)
ax2.set_title('QQ-Plot Gammaverteilung Geschwindigkeit [m/s]')

# Haupttitel hinzufügen
plt.suptitle("QQ-Plots für Geschwindigkeitsverteilungen in Zone 1", fontsize=16)


# Layout anpassen, um Überlappungen zu vermeiden
plt.tight_layout()

# Zeige den kombinierten Plot
plt.show()



# %%
dist_params = norm.fit(zone_1_cleaned['Speed_m/s'])

param_dict['speed_zone_1'] = ('norm', dist_params)
param_dict

# %% [markdown]
# ### 3.2.2 zone 1 mass

# %%
# Setzen Sie den Hintergrund
sns.set(style="whitegrid")
# density plot of zone 1 massen
sns.distplot(zone_1_cleaned['Mass_kg'])
# add title 
plt.title('Dichtekurve der Masse in Zone 1')
#X-Achse beschriften
plt.xlabel('Masse [kg]')


# %%
# sns boxplot for zone 1
sns.boxplot(data=zone_1_cleaned, x='Mass_kg')
# title of the plot
plt.title('Boxplot of zone 1 Mass_kg')

# %%
# fitter für zone 1 masse
# f = Fitter(zone_1_cleaned['Mass_kg'])
f = Fitter(zone_1_cleaned['Mass_kg'], distributions=get_common_distributions())
f.fit()
f.summary()

# %%
# Fit gamma distribution to data
dist_params = expon.fit(zone_1_cleaned['Mass_kg'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Mass_kg']), dist=expon, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot Exponentialverteilung Masse [kg]")
# Showing plot
plt.show()


# %%
# Fit gamma distribution to data
dist_params = gamma.fit(zone_1_cleaned['Mass_kg'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Mass_kg']), dist=gamma, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot gamma-Verteilung Mass_kg zone 1")
# Showing plot
plt.show()

# %%
#Zeige die QQ-Plots von den Verteilungen der Masse

# Vorbereitung der Grafik mit zwei Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Hintergrund der Grafik
fig.patch.set_facecolor('white')

# QQ-Plot für die Gammaverteilung
dist_params = gamma.fit(zone_1_cleaned['Mass_kg'])
probplot(np.array(zone_1_cleaned['Mass_kg']), dist=gamma, sparams=dist_params, plot=ax1)
ax1.set_title("QQ-Plot Gammaverteilung Masse [kg]")

# QQ-Plot für die Exponenzialverteilung
dist_params = expon.fit(zone_1_cleaned['Mass_kg'])
probplot(np.array(zone_1_cleaned['Mass_kg']), dist=expon, sparams=dist_params, plot=ax2)
ax2.set_title('QQ-Plot Exponentialverteilung Masse [kg]')

# Haupttitel hinzufügen
plt.suptitle("QQ-Plots für Masseverteilungen in Zone 1", fontsize=16)


# Layout anpassen, um Überlappungen zu vermeiden
plt.tight_layout()

# Zeige den kombinierten Plot
plt.show()

# %%
dist_params = gamma.fit(zone_1_cleaned['Mass_kg'])
param_dict['masse_zone_1'] = ('gamma', dist_params)
param_dict

# %% [markdown]
# ### 3.2.3 zone 1 time delta
# 

# %%
# density plot of zone 1 massen
sns.distplot(zone_1_cleaned['Time_Delta_hours'])
# add title 
plt.title('Density plot of zone 1 time deltas')

# %%
# sns boxplot for zone 1
sns.boxplot(data=zone_1_cleaned, x='Time_Delta_hours')
# title of the plot
plt.title('Boxplot of zone 1 Time_Delta_hours')

# %%
# fitter für zone 1 Time deltas
#f = Fitter(zone_1_cleaned['Time_Delta_hours'])
f = Fitter(zone_1_cleaned['Time_Delta_hours'], distributions=get_common_distributions())
f.fit()
f.summary()

# %%
# Fit gamma distribution to data
shape, loc, scale = gamma.fit(zone_1_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Time_Delta_hours']), dist=gamma, sparams=(shape, loc, scale), plot=plt)
# Adding title to plot
plt.title("QQ-Plot Gamma-Verteilung Time_Delta_hours")
# Showing plot
plt.show()

# %%
# Fit gamma distribution to data
dist_params = expon.fit(zone_1_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Time_Delta_hours']), dist=expon, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot Expon-Verteilung Time_Delta_hours")
# Showing plot
plt.show()

# %%
# Fit gamma distribution to data
dist_params = truncweibull_min.fit(zone_1_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Time_Delta_hours']), dist=truncweibull_min, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot Expon-Verteilung Time_Delta_hours")
# Showing plot
plt.show()

# %%
dist_params = expon.fit(zone_1_cleaned['Time_Delta_hours'])
param_dict['time_delta_zone_1'] = ('expon', dist_params)
param_dict

# %% [markdown]
# ### 3.3 zone 2

# %% [markdown]
# ### 3.3.1 zone 2 speed

# %%
# density plot of zone 2 speeds
sns.distplot(zone_2_cleaned['Speed_m/s'])
# add title 
plt.title('Density plot of zone 2 speeds')

# %%
# sns boxplot for zone 2
sns.boxplot(data=zone_2_cleaned, x='Speed_m/s')
# title of the plot
plt.title('Boxplot of zone 2 speed')

# %%
# fitter für zone 2 speed
#f = Fitter(zone_2_cleaned['Speed_m/s'])
f = Fitter(zone_2_cleaned['Speed_m/s'], distributions=get_common_distributions())
f.fit()
f.summary()

# %%
dist_params = gamma.fit(zone_2_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_2_cleaned['Speed_m/s']), dist=gamma, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title('QQ-Plot Gamma-Verteilung Speed km/h')
# Showing plot
plt.show()

# %%
dist_params= skewnorm.fit(zone_2_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_2_cleaned['Speed_m/s']), dist=skewnorm, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot skewnorm-Verteilung Speed km/h")
# Showing plot
plt.show()

# %%
dist_params = norm.fit(zone_2_cleaned['Speed_m/s'])
# Creating Q-Q-Plot
probplot(np.array(zone_2_cleaned['Speed_m/s']), dist=norm, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot norm-Verteilung Speed km/h")
# Showing plot
plt.show()

# %%
dist_params = norm.fit(zone_2_cleaned['Speed_m/s'])
param_dict['speed_zone_2'] = ('norm', dist_params)
param_dict

# %% [markdown]
# ### 3.3.2 zone 2 mass

# %%
# density plot of zone 2 mass
sns.distplot(zone_2_cleaned['Mass_kg'])
# add title 
plt.title('Density plot of zone 2 mass')

# %%
# sns boxplot for zone 2 mass
sns.boxplot(data=zone_1_cleaned, x='Mass_kg')
# title of the plot
plt.title('Boxplot of zone 2 Mass_kg')

# %%
# fitter für zone 2 Masse
#f = Fitter(zone_2_cleaned['Mass_kg'])
f = Fitter(zone_2_cleaned['Mass_kg'], distributions=get_common_distributions())
f.fit()
f.summary()

# %%
# Fit dweibull distribution to data
dist_params = dweibull.fit(zone_1_cleaned['Mass_kg'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Mass_kg']), dist=dweibull, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot dweibull-Verteilung Mass_kg zone 2")
# Showing plot
plt.show()


# %%
# Fit gamma distribution to data
dist_params = gamma.fit(zone_2_cleaned['Mass_kg'])
# Creating Q-Q-Plot
probplot(np.array(zone_2_cleaned['Mass_kg']), dist=gamma, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot gamma-Verteilung Mass_kg zone 2")
# Showing plot
plt.show()

# %%
dist_params = gamma.fit(zone_2_cleaned['Mass_kg'])
param_dict['masse_zone_2'] = ('gamma', dist_params)
param_dict

# %% [markdown]
# ### 3.3.3 zone 2 time delta

# %%
# density plot of zone 2 time delta
sns.distplot(zone_2_cleaned['Time_Delta_hours'])
# add title 
plt.title('Density plot of zone 2 time deltas')

# %%
# sns boxplot for zone 2 Time Deltas
sns.boxplot(data=zone_2_cleaned, x='Time_Delta_hours')
# title of the plot
plt.title('Boxplot of zone 2 Time_Delta_hours')

# %%
# fitter für zone 2 Time Deltas
#f = Fitter(zone_2_cleaned['Time_Delta_hours'])
f = Fitter(zone_2_cleaned['Time_Delta_hours'], distributions=get_common_distributions())
f.fit()
f.summary()

# %%
dist_params= gamma.fit(zone_2_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_2_cleaned['Time_Delta_hours']), dist=gamma, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot Gamma-Verteilung Time_Delta_hours zone 2")
# Showing plot
plt.show()

# %%
dist_params = expon.fit(zone_1_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Time_Delta_hours']), dist=expon, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot Expon-Verteilung Time_Delta_hours zone 2")
# Showing plot
plt.show()

# %%
dist_params = geninvgauss.fit(zone_1_cleaned['Time_Delta_hours'])
# Creating Q-Q-Plot
probplot(np.array(zone_1_cleaned['Time_Delta_hours']), dist=geninvgauss, sparams=dist_params, plot=plt)
# Adding title to plot
plt.title("QQ-Plot geninvgauss-Verteilung Time_Delta_hours zone 2")
# Showing plot
plt.show()

# %%
dist_params = expon.fit(zone_2_cleaned['Time_Delta_hours'])
param_dict['delta_zone_2'] = ('expon', dist_params)
param_dict

# %% [markdown]
# # 4 Distribution Params

# %%
param_dict

# %% [markdown]
# # Simulation Analysis

# %%
# read csv file 
data = pd.read_csv('../simulation/raw_rocks.csv')
#data = pd.read_csv('/Users/kimon/Downloads/output_500_000_000.csv')
data = data.rename(columns = {'mass_kg':'Mass_kg','speed_m/s':'Speed_m/s'})

data


# %%
data.describe()

# %%
data['zone'] == 2


# %%
# split data into zone 1 and zone 2
simulated_zone_1 = data[data['zone'] == 1]
# split data into zone 2
simulated_zone_2 = data[data['zone'] == 2]

#Delta Time Variable wiederherstellen
simulated_zone_1['Time_Delta_hours'] = simulated_zone_1['datetime'] - simulated_zone_1['datetime'].shift(1)
simulated_zone_1['Time_Delta_hours'] = simulated_zone_1['Time_Delta_hours'].fillna(0)

#Delta Time Variable wiederherstellen
simulated_zone_2['Time_Delta_hours'] = simulated_zone_2['datetime'] - simulated_zone_2['datetime'].shift(1)
simulated_zone_2['Time_Delta_hours'] = simulated_zone_2['Time_Delta_hours'].fillna(0)


# %%
simulated_zone_1


# %%
simulated_zone_2

# %% [markdown]
# ## Compare distributions Zone 1

# %%
# Setzen Sie den Hintergrund auf weiß mit Gitterlinien
sns.set(style="whitegrid")

# Erstellen Sie eine Figur mit zwei Spalten und drei Zeilen für Subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 21))  # 3 Zeilen, 2 Spalten

##### Dichtekurve Geschwindigkeit Zone 1
# Density plot for original data
sns.kdeplot(zone_1_cleaned['Speed_m/s'], label='Originaldaten', color='blue', ax=axs[0, 0], clip = (0, 18))


# Density plot for simulated data
sns.kdeplot(simulated_zone_1['Speed_m/s'], label='Simulationsdaten', color='orange', ax=axs[0, 0], clip = (0, 18))

# Set plot properties
axs[0, 0].set_title('Dichtekurven Geschwindigkeit Zone 1: Original- und Simulationsdaten')
axs[0, 0].set_xlabel('Geschwindigkeit [m/s]')
axs[0, 0].set_ylabel('Dichte')
axs[0, 0].legend()


##### CDF Geschwindigkeit Zone 1
# Sortieren der Daten
sorted_original_data = np.sort(zone_1_cleaned['Speed_m/s'])
sorted_simulated_data = np.sort(simulated_zone_1['Speed_m/s'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[0, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[0, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')
axs[0, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Geschwindigkeit Zone 1')
axs[0, 1].set_xlabel('Geschwindigkeit [m/s]')
axs[0, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[0, 1].legend()

##### Dichtekurve Masse Zone 1
# Density plot for original data
sns.kdeplot(zone_1_cleaned['Mass_kg'], label='Originaldaten', color='blue', ax=axs[1, 0], clip = (0, 8000))

# Density plot for simulated data
sns.kdeplot(simulated_zone_1['Mass_kg'], label='Simulationsdaten', color='orange', ax=axs[1, 0], clip = (0, 8000))
# Set plot properties
axs[1, 0].set_title('Dichtekurven Masse Zone 1: Original- und Simulationsdaten')
axs[1, 0].set_ylabel('Dichte')
axs[1, 0].set_xlabel('Masse [kg]')
axs[1, 0].legend()

##### CDF Masse Zone 1
# Sortieren der Daten
sorted_original_data = np.sort(zone_1_cleaned['Mass_kg'])
sorted_simulated_data = np.sort(simulated_zone_1['Mass_kg'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[1, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[1, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')

axs[1, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Masse Zone 1')
axs[1, 1].set_xlabel('Masse [kg]')
axs[1, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[1, 1].legend()

##### Dichtekurve Time Delta Zone 1
# Density plot for original data
sns.kdeplot(zone_1_cleaned['Time_Delta_hours'], label='Originaldaten', color='blue', ax=axs[2, 0], clip = (0, 300))

# Density plot for simulated data
sns.kdeplot(simulated_zone_1['Time_Delta_hours'], label='Simulationsdaten', color='orange', ax=axs[2, 0], clip = (0, 300))
# Set plot properties
axs[2, 0].set_title('Dichtekurven Delta-Zeit Zone 1: Original- und Simulationsdaten')
axs[2, 0].set_ylabel('Dichte')
axs[2, 0].set_xlabel('Delta-Zeit')
axs[2, 0].legend()

##### CDF Delta-Zeit Zone 1
# Sortieren der Daten
sorted_original_data = np.sort(zone_1_cleaned['Time_Delta_hours'])
sorted_simulated_data = np.sort(simulated_zone_1['Time_Delta_hours'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[2, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[2, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')
axs[2, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Delta-Zeit Zone 1')
axs[2, 1].set_xlabel('Delta-Zeit')
axs[2, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[2, 1].legend()

# Layout anpassen, um Überlappungen zu verhindern
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Zeigen Sie die kombinierten Plots an
plt.show()

# %% [markdown]
# ### Compare distributions Zone 2

# %%
# Setzen Sie den Hintergrund auf weiß mit Gitterlinien
sns.set(style="whitegrid")

# Erstellen Sie eine Figur mit zwei Spalten und drei Zeilen für Subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 21))  # 3 Zeilen, 2 Spalten

##### Dichtekurve Geschwindigkeit Zone 2
# Density plot for original data
sns.kdeplot(zone_2_cleaned['Speed_m/s'], label='Originaldaten', color='blue', ax=axs[0, 0], clip = (0, 70))

# Density plot for simulated data
sns.kdeplot(simulated_zone_2['Speed_m/s'], label='Simulationsdaten', color='orange', ax=axs[0, 0], clip = (0, 70))

# Set plot properties
axs[0, 0].set_title('Dichtekurven Geschwindigkeit Zone 2: Original- und Simulationsdaten')
axs[0, 0].set_xlabel('Geschwindigkeit [m/s]')
axs[0, 0].set_ylabel('Dichte')
axs[0, 0].legend()


##### CDF Geschwindigkeit Zone 2
# Sortieren der Daten
sorted_original_data = np.sort(zone_2_cleaned['Speed_m/s'])
sorted_simulated_data = np.sort(simulated_zone_2['Speed_m/s'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[0, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[0, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')
axs[0, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Geschwindigkeit Zone 1')
axs[0, 1].set_xlabel('Geschwindigkeit [m/s]')
axs[0, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[0, 1].legend()

##### Dichtekurve Masse Zone 2
# Density plot for original data
sns.kdeplot(zone_2_cleaned['Mass_kg'], label='Originaldaten', color='blue', ax=axs[1, 0], clip = (0, 1000))

# Density plot for simulated data
sns.kdeplot(simulated_zone_2['Mass_kg'], label='Simulationsdaten', color='orange', ax=axs[1, 0], clip = (0, 1000))
# Set plot properties
axs[1, 0].set_title('Dichtekurven Masse Zone 2: Original- und Simulationsdaten')
axs[1, 0].set_ylabel('Dichte')
axs[1, 0].set_xlabel('Masse [kg]')
axs[1, 0].legend()

##### CDF Masse Zone 2
# Sortieren der Daten
sorted_original_data = np.sort(zone_2_cleaned['Mass_kg'])
sorted_simulated_data = np.sort(simulated_zone_2['Mass_kg'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[1, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[1, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')

axs[1, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Masse Zone 2')
axs[1, 1].set_xlabel('Masse [kg]')
axs[1, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[1, 1].legend()

##### Dichtekurve Time Delta Zone 2
# Density plot for original data
sns.kdeplot(zone_2_cleaned['Time_Delta_hours'], label='Originaldaten', color='blue', ax=axs[2, 0], clip = (0, 600))

# Density plot for simulated data
sns.kdeplot(simulated_zone_2['Time_Delta_hours'], label='Simulationsdaten', color='orange', ax=axs[2, 0], clip = (0, 600))
# Set plot properties
axs[2, 0].set_title('Dichtekurven Delta-Zeit Zone 2: Original- und Simulationsdaten')
axs[2, 0].set_ylabel('Dichte')
axs[2, 0].set_xlabel('Delta-Zeit')
axs[2, 0].legend()

##### CDF Delta-Zeit Zone 2
# Sortieren der Daten
sorted_original_data = np.sort(zone_2_cleaned['Time_Delta_hours'])
sorted_simulated_data = np.sort(simulated_zone_2['Time_Delta_hours'])

# Kumulative Verteilungsfunktion
cumulative_distribution_original = np.linspace(0, 1, len(sorted_original_data))
cumulative_distribution_simulated = np.linspace(0, 1, len(sorted_simulated_data))

# Plot kumulative Verteilungen
axs[2, 1].plot(sorted_original_data, cumulative_distribution_original, label='Originaldaten', color='blue')
axs[2, 1].plot(sorted_simulated_data, cumulative_distribution_simulated, label='Simulationsdaten', color='orange')
axs[2, 1].set_title('Kumulative Verteilung von Original- und Simulationsdaten Delta-Zeit Zone 2')
axs[2, 1].set_xlabel('Delta-Zeit')
axs[2, 1].set_ylabel('Kumulative Wahrscheinlichkeit')
axs[2, 1].legend()

# Layout anpassen, um Überlappungen zu verhindern
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Zeigen Sie die kombinierten Plots an
plt.show()

# %% [markdown]
# ### Jährliche Wahrscheinlichkeit der Todesfälle pro simuliert nach Jahre

# %%
# Daten einlesen
data_conv = pd.read_csv('../data/convergence_simulation.csv')
data_conv


# %%
#Plot erstellen mit der Konvergenz der simulierten Daten
plt.figure(figsize=(12, 8))  # 
plt.scatter(data_conv['Simulierte Steinschläge'], data_conv['Wahrscheinlichkeit Todesfall'], color='red') 
plt.plot(data_conv['Simulierte Steinschläge'], data_conv['Wahrscheinlichkeit Todesfall'], color='red', linewidth=1)  

# Achsen beschriften
plt.xlabel('Simulierte Steinschläge')
plt.ylabel('Wahrscheinlichkeit der Todes')
plt.title('Jährliche Wahrscheinlichkeit der Todesfälle pro simulierte Jahre')

plt.show()


