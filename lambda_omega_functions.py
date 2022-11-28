import numpy as np
import time
from scipy import fftpack
import math
from scipy import signal
import sys
from matplotlib import pyplot as plt
import random as rd

Tmax = 30
dt = 0.01
threshold = 0.1

t = np.arange(0,Tmax+dt,dt, dtype = 'float')
t = list(t)
print(len(t))
flag = False
graph = False

def solution(random_params):
    y = random_params[0]
    w = random_params[1]
    a = random_params[2]
    b = random_params[3]


    
    dt = 0.01
    Fx = np.zeros(len(t));
    Fy = np.zeros(len(t))
    D = .0
    
    Fx[0] = 1;
    Fy[0] = 0;
    
    for j in range(0,len(t)-1):  
        eta = np.random.randn()  
        if Fx[j] > 3:
            return np.zeros(len(t))
            break  
        mod = abs(Fx[j])+abs(Fy[j]);
        kx1 = y*Fx[j]-w*Fy[j]-b*mod*Fx[j]-a*mod*Fy[j];
        ky1 = w*Fx[j]+y*Fy[j]+a*mod*Fx[j]-b*mod*Fy[j];    
        ax = Fx[j]+kx1*dt;         
        ax = ax+ math.sqrt(2*D*dt)*eta;   
        ay = Fy[j]+ky1*dt;
        amod = abs(ax)+abs(ay);
        kx2 = y*ax-w*ay-b*amod*ax-a*amod*ay;
        ky2 = w*ax+y*ay+a*amod*ax-b*amod*ay;

     
        Fx[j+1] = Fx[j]+(kx1+kx2)*dt/2;       
        Fx[j+1] = Fx[j+1]+math.sqrt(2*D*dt)*eta;
        Fy[j+1] = Fy[j]+(ky1+ky2)*dt/2;
    return Fx

def solution_noise(random_params, d):
    y = random_params[0]
    w = random_params[1]
    a = random_params[2]
    b = random_params[3]


    
    dt = 0.01
    Fx = np.zeros(len(t));
    Fy = np.zeros(len(t))
    D = d
    
    Fx[0] = 1;
    Fy[0] = 0;
    
    for j in range(0,len(t)-1):  
        eta = np.random.randn()  
        if Fx[j] > 3:
            return np.zeros(len(t))
            break  
        mod = abs(Fx[j])+abs(Fy[j]);
        kx1 = y*Fx[j]-w*Fy[j]-b*mod*Fx[j]-a*mod*Fy[j];
        ky1 = w*Fx[j]+y*Fy[j]+a*mod*Fx[j]-b*mod*Fy[j];    
        ax = Fx[j]+kx1*dt;         
        ax = ax+ math.sqrt(2*D*dt)*eta;   
        ay = Fy[j]+ky1*dt;
        amod = abs(ax)+abs(ay);
        kx2 = y*ax-w*ay-b*amod*ax-a*amod*ay;
        ky2 = w*ax+y*ay+a*amod*ax-b*amod*ay;

     
        Fx[j+1] = Fx[j]+(kx1+kx2)*dt/2;       
        Fx[j+1] = Fx[j+1]+math.sqrt(2*D*dt)*eta;
        Fy[j+1] = Fy[j]+(ky1+ky2)*dt/2;
    return Fx

def solution_org(random_params):
    y = random_params[0]
    w = random_params[1]
    a = random_params[2]
    b = random_params[3]
    r = math.sqrt(y/b)

    Fx = [0.0] * int(Tmax/dt + 1)
    Fy = [0.0] * int(Tmax/dt + 1)
    Fx[0] = 1

    D = 0.001 #magnitude of noise
    for j in range(len(t)-1):
        eta = np.random.randn()
        rn = math.sqrt(2*D*dt)*eta
        Fx[j]+=rn

        kx1 = (y - b*r**2)*Fx[j] - (w+a*r**2)*Fy[j];
        ky1 = (w+a*r**2)*Fx[j] + (y-b*r**2)*Fy[j];

        ax = Fx[j] + kx1*dt + rn;
        ay = Fy[j] + ky1*dt + rn;

        kx2 = (y - b*r**2)*ax - (w+a*r**2)*ay;
        ky2 = (w+a*r**2)*ax + (y-b*r**2)*ay;

        Fx[j+1] = Fx[j] + (kx1+kx2)*dt/2 + rn;
        Fy[j+1] = Fy[j] + (ky1+ky2)*dt/2 + rn;
    Fx = [ round(elem, 4) for elem in Fx ]
    #Fy = [ round(elem, 4) for elem in Fy ]
    plt.plot(t,Fx)
    return Fx

def calculate_fitness(random_params,Fx):
    Fx_guess = solution_noise(random_params)
    fit3 = 0

    for i in range(len(t)):
        fit3 += abs(Fx_guess[i] - Fx[i])**2
        
    return [fit3/len(t)]
            
def calculate_fitness_org(random_params, Fx): 
    #original copy of calculate_fitness()
    #find peak indices for the ground truth model Fx
    peak_indices_truth = fpeaks(Fx)
    period_truth = find_period(peak_indices_truth, t, Fx)
    amplitude_truth = find_amplitude(Fx)
    #frequency = find_freq(period)

    Fx_guess = solution(random_params)
    peak_indices_guess = fpeaks(Fx_guess)
    period_guess = find_period(peak_indices_guess, t, Fx_guess)
    amplitude_guess = find_amplitude(Fx_guess)
    #frequency_guess = find_freq(period_guess)

    period = abs(period_truth-period_guess)

    if(flag == True):
        print("Ground truth period: " + str(round(period_truth,3)) + "\t" +
                        " Fake period: " + str(round(period_guess,3)))

    amplitude = abs(amplitude_truth-amplitude_guess)
    if(flag == True):
        print("Ground truth amplitude: "+str(round(amplitude_truth,3)) + "\t" +
                    " Fake amplitude: "+str(round(amplitude_guess,3)))

    fit3 = 0

    for i in range(len(t)):
        fit3+= abs(Fx_guess[i] - Fx[i])
    #fit4 = abs(frequency-frequency_guess)
    
    
    return [round(period, 4), round(amplitude, 4)]

def dominates(val1, val2, num_of_objectives):
    for coord in range(len(val1)-num_of_objectives, len(val1)):
        #check if it is less than because the smaller value is better fitness
        #print(coord)
        #print(val1)
        #print(val2)
        if(val1[coord] <= val2[coord]):
            continue
        return False
    return True

def fast_nds(p, num_of_objectives):
    all_fronts = []
    f_count = 1
    Fi = []
    Sp_Np_all = {}
    for soln1 in p:
        Sp = [] #set of solutions dominated by soln1
        Np = 0 #number of solutions which dominate soln1
        for soln2 in p:
            if(soln1 != soln2):
                if(dominates(p[soln1], p[soln2], num_of_objectives)):
                    Sp.append(soln2)
                elif(dominates(p[soln2], p[soln1], num_of_objectives)):
                    #print(soln1)
                    Np+=1
        if(Np == 0):
            Fi.append(soln1)
        Sp_Np_all[soln1] = [Sp, Np]
        #fronts[soln1]
    #all_fronts.append(Fi)
    #print(Fi)
    #print(Sp_Np_all)

    while(Fi):
        all_fronts.append(Fi)
        Q = []
        for p in Fi:
            for q in Sp_Np_all[p][0]:

                Sp_Np_all[q][1] -=1
                Nq = Sp_Np_all[q][1]
                if(Nq == 0):
                    Q.append(q)
        Fi = Q

    #print("All fronts: ", all_fronts)
    return all_fronts

def fpeaks(X):
    vals = signal.find_peaks(X, prominence=[0.1,None])
    return np.array(vals).tolist()[0].tolist()

def find_period(pks, t, Fx):
    c = 0
    p = 0
    prev = 0
    if(graph):
        plt.figure()
        plt.plot(t, Fx)
        plt.xlim(0, 20)
        #plt.ylim((-5,5))
        plt.title("period")
    if(len(pks)==1):
        return sys.maxsize

    for i in range(len(pks)-1):
        if(graph):
            plt.plot(t[pks[i]], Fx[pks[i]], 'or')
        p = abs(t[pks[i+1]] - t[pks[i]])

        if(flag):
            print("period ", prev, p)
        if(abs(p-prev) < threshold):
            return p
        elif(i == len(pks) - 2):
            break
        else:
            prev = p
    if(abs(p-prev) > threshold):
        #print("maxing out ", abs(p-prev))
        p = sys.maxsize
    return p

def find_freq(period):
    if(period == 0):
        return 1000
    return 1/period


def find_amplitude(Fx):
    Fx_flip = [0.0] * len(Fx)
    for i in range(len(Fx)):
        Fx_flip[i] = Fx[i] * (-1)

    troughs = fpeaks(Fx_flip)
    pks = fpeaks(Fx)

    if(len(pks) >= 1 and len(troughs) >= 1):
        prev = abs(Fx[pks[0]] - Fx[troughs[0]])/2
    else:
        return sys.maxsize

    i = 0
    prev = 0
    #lst = ['ob', 'vr', 'xg', '+b', 'Dr', '*g', '|b', 'Pr']
    lst = ['ob']
    if(graph):
        plt.figure()
        plt.title("amplitude")
        plt.plot(t, Fx)
        plt.xlim((0,15))
        plt.ylim((-3, 3))

    while( i<len(pks) and i<len(troughs)):
        amplitude = abs(Fx[pks[i]] - Fx[troughs[i]])/2
        if(flag):
            print("amplitude ", amplitude, prev)
        if(graph):
            plt.plot(t[pks[i]], Fx[pks[i]], lst[0])
            plt.plot(t[troughs[i]], Fx[troughs[i]], lst[0])

        k = abs(amplitude - prev)
        if(k <= threshold):
            break
            return amplitude
        elif(i == len(pks)-1 or i == len(troughs)-1):
            pass
            #amplitude = sys.maxsize
        else:
            prev = amplitude
        i+=1
    if(abs(prev-amplitude) > threshold):
        pass
        # amplitude = sys.maxsize
        # amplitude = float('inf')

    return amplitude
