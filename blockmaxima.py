import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import simulation

u1, u2 = simulation.simu_clayton(num=2500, theta=0.6)
from scipy.stats import norm
#apply distribution.ppf to transform uniform margin to the desired distribution in scipy.stats
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)


# Block maxima 
block_lenght = 20 #number of trading days in one month
# Split data by block of equal lenght
u1_splitted = [u1[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u1) + block_lenght - 1) // block_lenght )]  
u2_splitted = [u2[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u2) + block_lenght - 1) // block_lenght )]  

block_maxima_u1 = []
block_maxima_u2 = []
for block_u1, block_u2 in zip(u1_splitted, u2_splitted):
    max_block_u1 = max(block_u1)
    max_block_u2 = max(block_u2)
    block_maxima_u1.append(max_block_u1)
    block_maxima_u2.append(max_block_u2)

plt.scatter(u1, u2, color="black", alpha=0.8)
plt.scatter(block_maxima_u1, block_maxima_u2, color="red", alpha=0.8)
plt.show()
