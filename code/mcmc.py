import numpy as np
import pandas as pd
import emcee
import machine_learning
import priors
from multiprocessing import cpu_count, active_children
from multiprocessing.pool import Pool
import time
import sys 

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

def log_prior(theta):
    """
    Compute the prior probability of a given set of parameters theta

    Parameters
    ----------
    theta : `~numpy.ndarray`
        A (6,) array which contains m1i, m2i, ai, z, x and alphaCE of a walker.

    Returns
    -------
    The probability that this prior is accepted
    """

    m1, m2, a, xfer, alpha, z = theta
    eps = 1E-3 
    if m1 < m2:
        return -np.inf
    if 19.-eps < m1 < 66.-eps:
        proba_m1 = -2.3*np.log10(m1)
    else : 
        proba_m1 = -np.inf
        
    if 14.-eps < m2 < 50.+eps:
        proba_m2 = -2.3*np.log10(m2)
    else :
        proba_m2 = -np.inf
        
    if 19.-eps < a < 300.+eps :
        proba_a = 0.0
    else :
        proba_a = -np.inf
    
    if -5.-eps < z < -1.83+eps:
        proba_z = 0.0
    else :
        proba_z = -np.inf
        
    if 0.2-eps < xfer < 0.8+eps :
        proba_xfer = 0.0
    else : 
        proba_xfer = -np.inf
        
    if  1.-eps < alpha < 2.+eps :
        proba_alpha = 0.0
    else :
        proba_alpha = -np.inf
    return proba_m1+proba_m2+proba_a+proba_z+proba_xfer+proba_alpha

def log_likelihood(kernel, m1, m2):
    """
    Compute the likelihood of a given final state to cause the GW event

    Parameters
    ----------
    kernel : 
        The distribution of LIGO/Virgo posterior in masses given a GW event
    m1 :
        The mass of the first compact object
    m2 :
        The mass of the second compact object

    Returns
    -------
    The probability that a binary (m1, m2) caused the GW event
    """

    proba = kernel.evaluate([m1, m2])
    if proba>0:
        return np.log10(proba)
    else:
        return -np.inf
   
def likelihood_position(theta, kernel, model, data, target):
    """
    Compute the likelihood of a given initial bianry stellar system to create a DCO that will cause the GW event

    Parameters
    ----------
    kernel : 
        The distribution of LIGO/Virgo posterior in masses given a GW event
    theta :
        The initial parameters of the binary systems

    Returns
    -------
    The probability that a binary stellar system created a DCO that will cause the GW event

    Calls
    -----
    log_likelihood
    """

    model_1 = model.fit(data.values, target['Final mass 1st CO'])
    m1f = model_1.predict(theta.reshape(1,-1))
    model_2 = model.fit(data.values, target['Final mass 2nd CO'])
    m2f = model_2.predict(theta.reshape(1,-1))
    m1_f = np.max([m1f, m2f])
    m2_f = np.min([m1f, m2f])
    ll = log_likelihood(kernel, m1_f, m2_f)
    return ll

def log_probability(theta, kernel, model, data, target):
    """
    Compute the probability of a given initial binary stellar system was formed and created a DCO
    that will cause the GW event

    Parameters
    ----------
    kernel : 
        The distribution of LIGO/Virgo posterior in masses given a GW event
    theta :
        The initial parameters of the binary systems

    Returns
    -------
    The probability that a binary stellar system was formed and created a DCO that will cause the GW event

    Calls
    -----
    log_prior, log_likelihood
    """
    model.fit(data.values, target['Final mass 1st CO'])
    m1f = model.predict(theta.reshape(1,-1))
    model= model.fit(data.values, target['Final mass 2nd CO'])
    m2f = model.predict(theta.reshape(1,-1))
    # Vérifier ces deux lignes
    m1_f = np.max([m1f, m2f])
    m2_f = np.min([m1f, m2f])
    lp = log_prior(theta)
    ll = log_likelihood(kernel, m1_f, m2_f)
    if not np.isfinite(lp):
        return -np.inf
    lprob = lp + ll
    return lprob

def choose_position_i(n):
    """
    choose initial positions for n walkers

    Parameters
    ----------
    n : The number of walkers

    Returns
    -------
    pos : n walkers initialised at some position in the phase space
    """
    pos_m1 = np.random.randint(19, 50, n)
    pos_m2=99*np.ones(n)

    for i in range(0, len(pos_m1)):
        while pos_m2[i]>=pos_m1[i]:
            pos_m2[i] = np.random.randint(14, 40)
            if pos_m1[i]>= 2*pos_m2[i]:
                pos_m2[i]= 99
                
    
    pos_a = np.power(10, np.random.random(n)+1.3)
    pos_z = np.random.randint(-5, -1, n)
    pos_x = 0.2*np.random.randint(1, 5, n)
    pos_alpha = np.random.random(n)+1


    pos = np.array([pos_m1, pos_m2, pos_a, pos_x, pos_alpha, pos_z])
    pos = np.around(np.transpose(pos), 6)
    
    return pos

def check_walkers(kernel, model, data, target):
    """
    Gather the most likely walker in an array of walkers, to create a final set of 30 walkers

    Parameters
    ----------
    kernel : The distribution of LIGO/Virgo posterior in masses given a GW event

    Returns
    -------
    pos : 30 walkers initialised at some position in the phase space

    Calls
    -----
    choose_position, lieklihood_position
    """
    n_walkers=20
    pos=[]
    for i in range(0, n_walkers):
        pos_i = choose_position_i(20)
        pos_i_ll = []
        for k in range(0, len(pos_i)):
            pos_i_ll.append(float(likelihood_position(pos_i[k], kernel, model, data, target)))
        maxi_ll = max(pos_i_ll)
        max_index = pos_i_ll.index(maxi_ll)
        pos.append(pos_i[max_index])
    pos=np.array(pos)
    return(pos)

def saving(sampler, string):
    np.save('samples_%s.npy' % string, sampler.get_chain())
    np.save('blobs_%s.npy' % string , sampler.get_blobs())
    np.save('last_%s.npy' % string, sampler.get_last_sample())
    np.save('log_prob_%s.npy' % string, sampler.get_log_prob())
    np.save('auto_corr_%s.npy' % string, sampler.get_autocorr_time(quiet=True))
    np.save('acc_ratio_%s.npy' % string, sampler.acceptance_fraction)
    return 0


def emcee_run(n_steps, kernel, model, data, target):
    pos = check_walkers(kernel, model, data, target)
    nwalkers, ndim = pos.shape
    
    with Pool() as pool:
        print(pool)
        # report the number of processes in the pool
        print(pool._processes)
        # report the number of active child processes
        children = active_children()
        print(len(children))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        moves=emcee.moves.StretchMove(a=2),
                                        args=(kernel,model, data, target), pool=pool)
        start = time.time()
        sampler.run_mcmc(initial_state=pos, nsteps=n_steps, progress=True)
        
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    return sampler

ID = sys.argv[1]
print(ID)
file_data = '/scratch/marchior/run_MCMC/%s/data.h5' % ID
file_target = '/scratch/marchior/run_MCMC/%s/target.h5' % ID
file_prior = "/scratch/marchior/run_MCMC/%s/masses_%s.h5" % (ID, ID)

def main():
    data, target, model_ML = machine_learning.main(file_data, file_target)
    density=priors.main(file_prior)
    sampler_end = emcee_run(200000, density, model_ML, data, target)
    saving(sampler_end, '%s' % ID)
    return 0

if __name__ == '__main__':
    main()
