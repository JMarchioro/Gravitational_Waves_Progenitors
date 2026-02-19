import numpy as np
import pandas as pd
import scipy.stats

# file_prior = "/work/marchior/MCMC/data/masses_GW191216_213338.h5"
def main(file_prior):
    df = pd.read_hdf(file_prior, 's')
    density = scipy.stats.gaussian_kde(df.T)
    return(density)
