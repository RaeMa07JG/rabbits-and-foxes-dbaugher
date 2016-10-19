import numpy as np
import random

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04

'''Peak selection adopted from Greg Zaylor's notebook - thank you Greg'''
#create list to store values at peaks for foxes and times:
foxes = [[0,0],[0,0]]
#define number of iterations
Trials = 100

def experiment(Trials):
    global zeropopulation
    zeropopulation=0
    R_0 = 400.
    F_0 = 200.
    end = 600. #days

    for i in range(Trials):
        #run the KMC
        R = R_0
        F = F_0
        time = 0
        solution_exp = np.zeros((1,2))
        solution_exp[0] = 0, F_0
        while time < end:
            Rb = k1*R
            Rd = k2*R*F
            Fb = k3*R*F
            Fd = k4*F
            Rcum = Rb + Rd + Fb + Fd
            u = random.uniform(0,1)
            event = u * Rcum
            if Rb > event:
                R = R + 1
            elif Rd + Rb > event >= Rb:
                R = R - 1
            elif Fb + Rd + Rb > event >= Rd + Rb:
                F = F + 1
            else:
                F = F - 1

            time = time + 1/Rcum * np.log(1/u)
            solution_exp = np.append(solution_exp,[[time,F]],axis=0)

            if F==0:
                zeropopulation = zeropopulation + 1
                break

        time_exp = solution_exp[:,0]

        filtered = solution_exp[:,1] * (time_exp>200) * (solution_exp[:,1]>200)
        c = filtered.max()
        t = time_exp[filtered.argmax()]
        #if 2nd peak exists, add number of foxes and time it occurs to storage list. only return first peak if peak repeats itself
        if c>200:
            foxes[1].append(c)
            foxes[0].append(t)

    return foxes

experiment(Trials)

print("The average of the 2nd peak is", round(sum(foxes[1])/float((len(foxes[1])-1))), "foxes, which occurs at ", round(sum(foxes[0])/float((len(foxes[0])-1))), "days.")
print("The probability that no 2nd peak forms (and all the foxes die out before 600 days) is ",100*zeropopulation/Trials,"%")

time_75, time_25 = np.percentile(foxes[0], [75 ,25])
fox_75, fox_25 = np.percentile(foxes[1], [75 ,25])
print("The IQR of foxes is", round(fox_25), "to", round(fox_75))
print("The IQR of times is", round(time_25), "to", round(time_75))