
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt


class gpd_pot():
    """
        
    """
    fit = ["mle", "mom", "pwm"]

    def __init__(self, data, tu, fit):
        self.data = data
        self.n = len(data)
        self.u = sorted(data)[int(self.n * tu)]
        self.values = (data.loc[data >= self.u]-self.u).values
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

        def fit_mom(self):
            a0 = self.values.mean()
            a1 = sum([i**2 for i in self.values])/len(self.values)
            Beta = (a0*a1)/(2*a1-2*a0**2)
            Xi = 0.5-(a0**2/(2*a1-2*a0**2))
            return Beta, Xi

        def fit_pwm(self):
            w0 = self.values.mean()
            w1 = sum([ (1- (sorted(self.values).index(x)-0.35)/len(self.values) )*x for x in sorted(self.values)])/len(self.values)
            Beta = (2*w0*w1)/(w0-2*w1)
            Xi = 2-(w0/(w0-2*w1))
            return Beta, Xi

        if fit == "mle":
            self.Beta, self.Xi = fit_mle(self)
        elif fit == "mom":
            self.Beta, self.Xi = fit_mom(self)
        elif fit == "pwm":
            self.Beta, self.Xi = fit_pwm(self)
    
    def quantile(self, q):
        return  (self.Beta/self.Xi)*( ((1-q) )**(-self.Xi) -1  )

    def varq(self, q):
        return  self.u+(self.Beta/self.Xi)*(((self.n/self.n_u)*(1-q))**(-self.Xi)-1)

    def esq(self, q):
        return (self.u+(self.Beta/self.Xi)*(((self.n/self.n_u)*(1-q))**(-self.Xi)-1))/(1-self.Xi)+(self.Beta-self.Xi*self.u)/(1-self.Xi)

    def qq_plot(self):
        ecdf = ECDF(self.values)
        observed_quantiles = sorted(self.values)
        theorical_quantiles = [self.quantile(q=ecdf(x)) for x in observed_quantiles] 

        x = np.linspace(min(self.values), max(self.values), 10)
        plt.plot(x, x, '-', color='red')
        plt.plot(observed_quantiles, theorical_quantiles, '.', color='black')
        plt.show()

    def CDF(self, x):
        if self.Xi ==0:
            return 1-np.exp(-x/self.Beta)
        else:
            return 1-(1+(self.Xi/self.Beta)*x)**(-1/self.Xi)
    
    def pp_plot(self):

        ecdf = ECDF(self.values)
        observed_cdf = [ecdf(x) for x in sorted(self.values)] 
        theorical_cdf = [self.CDF(x) for x in sorted(self.values)] 

        x = np.linspace(0, 1, 10)
        plt.plot(x, x, '-', color='red')
        plt.plot(observed_cdf, theorical_cdf, '.', color='black')
        plt.show()

def mean_exc(data):
    nb_sep = 100
    list_tu = [0.2*(i/nb_sep) for i in range(1,nb_sep)]
    mean_exc, conf_low, conf_high = [], [], []

    for tu_ in list_tu:
        u = sorted(data)[int(len(data) * (1-tu_))]
        values = ((data.loc[data >= u])).values
        mean_ = values.mean()-u
        std_ = values.std()
        mean_exc.append(mean_)
        conf_low.append(mean_-std_)
        conf_high.append(mean_+std_)

    plt.plot(list_tu, mean_exc, 'o', color='red')
    plt.fill_between(list_tu, conf_low, conf_high, alpha = 0.4)
    plt.show()