# Extreme-Risk
Using Extreme Value Theory (EVT) to Estimate Value-at-Risk (VaR) and Expected shortfall (ES)
The Peaks-Over-Threshold (POT) model of extreme value theory, and GPD distribution give more accurate description on tail distribution of financial losses.

## Simple Usage

```python
import pandas as pd
import numpy as np
import pot


df = pd.read_csv("data/msci1.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
df = df.dropna()


data = df["UK"]

print(data)

gpd = pot.gpd_pot(data, tu=0.1)

param = gpd.fit_mle()

varq = gpd.varq(Beta=param[0], Xi=param[1], q=0.99)
print(varq)
esq = gpd.esq(Beta=param[0], Xi=param[1], q=0.99)
print(esq)

```