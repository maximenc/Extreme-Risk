import pandas as pd
import numpy as np
import pot


df = pd.read_csv("data/msci.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
df = df.dropna()

data = -df["Italy"]
#US	UK	Switzerland	Sweden	Spain	Singapore	Norway	Netherlands	Japan	Italy



fitted_gpd = pot.gpd_pot(data, tu=0.95, fit="mle")
print(fitted_gpd.Beta, fitted_gpd.Xi)
fitted_gpd = pot.gpd_pot(data, tu=0.95, fit="mom")
print(fitted_gpd.Beta, fitted_gpd.Xi)
fitted_gpd = pot.gpd_pot(data, tu=0.95, fit="pwm")
print(fitted_gpd.Beta, fitted_gpd.Xi)

print(fitted_gpd.quantile(q=0.99))

pot.mean_exc(data)

fitted_gpd.qq_plot()
fitted_gpd.pp_plot()



