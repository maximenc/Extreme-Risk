
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class gpd_pot():
    """
        
    """
    def __init__(self, data, tu):
        self.data = data
        self.n = len(data)
        self.u = sorted(data)[int(self.n * tu)] 
        self.values = (-(data.loc[data <= self.u]-self.u)).values
        self.n_u = len(self.values)

    def fit_mle(self):

        def log_likelihood(param):
            log_l = -self.n_u*np.log(param[1]) - (1/param[0] + 1 )*sum([ np.log( 1 + (param[0]/param[1])*i) for i in self.values ])
            return -log_l

        bounds_param = ((0+1e-6,None), (0+1e-6,None))
        theta_start = (np.array(0), np.array(1))

        opti_method = 'L-BFGS-B'
        results = minimize(log_likelihood, theta_start, method=opti_method, bounds=bounds_param) #options={'maxiter': 300})#.x[0]

        Beta = results.x[1]
        Xi = results.x[0]

        return Beta, Xi

    def varq(self, Beta, Xi, q):
        return self.u + (Beta/Xi)*( ( (self.n/self.n_u)*(1-q) )**(-Xi) -1  )

    def esq(self, Beta, Xi, q):
        return (self.u + (Beta/Xi)*( ( (self.n/self.n_u)*(1-q) )**(-Xi) -1  )) / (1- Xi) + (Beta - Xi*self. u)/(1-Xi)
