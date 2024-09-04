import matplotlib
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
import math

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.antialiased'] = True
pd.set_option('display.float_format', '{:.4f}'.format)

df = pd.read_excel("data/DryBeanDataSet.xlsx")
df['Extent'] = df['Extent'].replace("?", np.nan)
df['Compactness'] = df['Compactness'].replace("?", np.nan)
df['ShapeFactor6'] = df['ShapeFactor6'].replace("?", np.nan)

df = df.drop(['Class', 'Sort order'], axis=1)

variables = df.columns

num_plots = len(variables)
#cols = math.ceil(math.sqrt(num_plots))
cols = 3
rows = math.ceil(num_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(8, 12))
axes = axes.flatten()
for i, var in enumerate(variables):
    sns.violinplot(ax=axes[i], x=var, data=df)
    axes[i].set_xlabel(f"{var}")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Distribution of features", fontsize=10)
plt.tight_layout()


plt.savefig('test.pdf', format='pdf')





















