import torch
import multiprocessing
import pandas as pd

import time
from scipy import fftpack
import math
from scipy import signal 
import sys
from matplotlib import pyplot as plt
import random as rd
from sbi import utils as utils
from sbi import analysis as analysis 
from sbi.inference.base import infer
from ipywidgets import *
import pickle
import numpy as np
from lambda_omega_functions import solution_noise

Tmax = 30
dt = 0.01
threshold = 0.1

t = np.arange(0,Tmax+dt,dt, dtype = 'float')
t = list(t)

exp = "_Experiment_11,_Trial_3"


pickleBool = True #false if loading from pickle
pickFile = "C:\\Users\\leder\\Anaconda3\\envs\\sbi_env\\MyScripts\\pickleJar" + exp

def sol(random_params):
    y = random_params[0]
    w = random_params[1]
    a = random_params[2]
    b = random_params[3]

    
    return solution_noise([y,w,a,b], 0)

#print("Hello world")
def calculate_summary_statistics(trace): #function will return the summary statistics for a certain simulation, currently only calculates freq and amplitude. What other metrics could be used that are applicable to lw model?  
    pks,pksDict = signal.find_peaks(trace, prominence=[2,None])
    freq = 1/np.mean(np.diff(pks))*dt
    amp = np.mean(pksDict['prominences']/2)
    return freq, amp

def simulation_wrapper(params):
    ######################################################################################
    order = [0,1,2,3] #this line is changed when different params are estimated for; [0,3] correlates to lambda and beta
    inp = [1,1,1,1]
    for elem in params:
        inp[order[0]] = elem
        order.pop(0)
    trace = sol(inp)
    summstats = calculate_summary_statistics(trace)
    return summstats

######################################################################################
dimensions = 4
upper = 2
prior = utils.BoxUniform(low=0.01*torch.ones(dimensions), high = upper*torch.ones(dimensions)) #creates prior distribution of parameters, not sure if this is a uniform distribution from 0 to upper and what torch.ones() does

observation = solution_noise([1,1,1,1], .04)
observation_summary_statistics = calculate_summary_statistics(observation)
true_params = np.array([1,1,1,1])


if(pickleBool):
    posterior = infer(simulation_wrapper, prior, method = 'SNPE', num_simulations =2000, num_workers = 4)
else:
    infile = open(pickFile, 'rb')
    posterior = pickle.load(infile)
    infile.close()




samples = posterior.sample((2000,), 
                           x=observation_summary_statistics, sample_with_mcmc = True)


#HBox(children=(FloatProgress(value=0.0, description='Drawing 10000 posterior samples', max=10000.0))) #not sure what this does, or if its necessary

######################################################################################
# fig, axes = analysis.pairplot(samples,
#                            limits=[[0, upper], [0, upper]],
#                            ticks=[[0, upper], [0, upper]],
#                            figsize=(5,5),
#                            points=true_params,
#                            points_offdiag={'markersize': 6},
#                            points_colors='r')   

# fig, axes = analysis.pairplot(samples,
#                            limits=[[0, upper], [0, upper],[0,upper]],
#                            ticks=[[0, upper], [0, upper], [0,upper]],
#                            figsize=(5,5),
#                            points=true_params,
#                            points_offdiag={'markersize': 6},
#                            points_colors='r')               


fig, axes = analysis.pairplot(samples,
                           limits=[[0, upper], [0, upper],[0, upper],[0, upper]],
                           ticks=[[0, upper], [0, upper],[0, upper],[0, upper]],
                           figsize=(5,5),
                           labels = ["\u03BB", "\u03C9", "\u03B1", "\u03B2"],
                           points=true_params,
                           points_offdiag={'markersize': 6},
                           points_colors='r')

labels = [" 0               2\n\u03BB", " 0               2\n\u03C9", " 0               2\n\u03B1", " 0               2\n\u03B2"]
count = 0

for axis in axes[:-1]:
    print(axis[0])

    #axis[len(axis) - 1].set_yticks([0, upper])
    axis[len(axis) - 1].set_ylabel(labels[count])
    axis[len(axis) - 1].yaxis.set_label_position("right")
    #axis[len(axis) - 1].yaxis.tick_right()
    count+=1


if(pickleBool):
    outfile = open(pickFile, "wb")
    pickle.dump(posterior, outfile)
    outfile.close()
