import pandas as pd
import numpy as np
from bivariate import copula, estimation, simulation

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox

df = pd.read_csv("data/msci1.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
df = df.dropna()
print(df.columns.values)


# 'US','UK','Switzerland','Sweden','Singapore', 'Norway','Netherlands', 
# 'Japan', 'Italy', 'HongKong', 'Germany', 'France', 'Denmark', 'Canada',
# 'Austria', 'Australia']

asset1 = "France"
asset2 = "Italy"
data = df[[asset1,asset2]]
fig, ax = plt.subplots()
ax.set_xlim(-0.16, 0.16)
ax.set_ylim(-0.16, 0.16)
plt.scatter(data.iloc[:,0],data.iloc[:,1], marker='o', color="black", alpha=0.8)
plt.xlabel(asset1)
plt.ylabel(asset2)

r = str( asset1 +"/" + asset2 )
#ax.add_artist(AnchoredText(r, loc=9, borderpad=-2.5, frameon=False, prop=dict(fontweight="bold",fontsize=12)))
ax.add_artist(AnchoredText("correlation", loc=9, borderpad=-1.5, frameon=False, prop=dict(fontweight="bold",fontsize=12)))
r = np.corrcoef(data.loc[:,asset1], data.loc[:,asset2])[0][1]
r = str("Corr = " + "{:.2f}".format(r) )
ax.add_artist(AnchoredText(r, loc=4))
plt.subplots_adjust(left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.2, hspace=0.2)
plt.show()

######################################################
fig, ax = plt.subplots()
ax.set_xlim(-0.16, 0.16)
ax.set_ylim(-0.16, 0.16)
plt.scatter(data.iloc[:,0],data.iloc[:,1], marker='o', color="black", alpha=0.8)
plt.xlabel(asset1)
plt.ylabel(asset2)

u = 0.99
quantile1_u = data[asset1].quantile(u)
quantile2_u = data[asset2].quantile(u)
cond_uu = (data[asset1]>quantile1_u) & (data[asset2]>quantile2_u)
r_uu = np.corrcoef(data.loc[cond_uu,asset1], data.loc[cond_uu,asset2])[0][1]
r_uu = str("Corr UU = " + "{:.2f}".format(r_uu) )
ax.add_artist(AnchoredText(r_uu, loc=1))

l = 0.01
quantile1_l = data[asset1].quantile(l)
quantile2_l = data[asset2].quantile(l)
cond_ll = (data[asset1]<quantile1_l) & (data[asset2]<quantile2_l)
r_ll = np.corrcoef(data.loc[cond_ll,asset1], data.loc[cond_ll,asset2])[0][1]
r_ll = str("Corr LL = " + "{:.2f}".format(r_ll) )
ax.add_artist(AnchoredText(r_ll, loc=3))

plt.vlines(quantile1_l, -0.16, quantile2_l, colors='r')
plt.vlines(quantile1_u, quantile2_u, 0.16, colors='r')
plt.hlines(quantile2_l, -0.16, quantile1_l, colors='r')
plt.hlines(quantile2_u, quantile1_u, 0.16, colors='r')

r = str("Correlation " + asset1 +"/" + asset2 + " = " + "{:.2f}".format(np.corrcoef(data.iloc[:,0], data.iloc[:,1])[0][1]))
r = r
ax.add_artist(AnchoredText("Quantile correlation (q=0.01)", loc=9, borderpad=-1.5, frameon=False, prop=dict(fontweight="bold",fontsize=12)))
plt.subplots_adjust(left=0.13, bottom=0.11, right=0.98, top=0.94, wspace=0.2, hspace=0.2)
plt.show()

