import pandas as pd
import numpy as np
import pot


df = pd.read_csv("data/msci1.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
df = df.dropna()


data = df["US"]
#US	UK	Switzerland	Sweden	Spain	Singapore	Norway	Netherlands	Japan	Italy

print(data)

from statsmodels.distributions.empirical_distribution import ECDF





gpd = pot.gpd_pot(data, tu=0.05)
print(gpd.u)
print(gpd.values)


param = gpd.fit_mle()

ecdf = ECDF(gpd.values)
print(gpd.values)

quantile_i = [gpd.varq(Beta=param[0], Xi=param[1], q=ecdf(x)) for x in sorted(gpd.values)] 
sorted(gpd.values)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()

x = np.linspace(min(gpd.values), max(gpd.values), 10)
plt.plot(x, x, '-', color='red')
plt.plot(sorted(gpd.values), quantile_i, '.', color='black')

plt.show()


"""
varq = gpd.varq(Beta=param[0], Xi=param[1], q=0.99)
print(varq)
esq = gpd.esq(Beta=param[0], Xi=param[1], q=0.99)
print(esq)
"""



print(param)

param = gpd.fit_mom()

print(param)

param = gpd.fit_mpwm()

print(param)