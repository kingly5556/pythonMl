import numpy as np
import matplotlib.pyplot as plt

# mu,sigma = 0,1
# x = np.linspace(-4,4,100)
# y = (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-0.5*((x - mu)/sigma)**2)
# plt.plot(x,y)
# plt.show()

# p = 0.5
# plt.bar([0,1],[1-p,p],color ="blue")
# plt.xticks([0,1],labels=["F","W"])
# plt.show()

from scipy.stats import binom,poisson

# n,p = 10,0.5
# x = np.arange(0,n+1)
# y= binom.pmf(x,n,p)
# plt.bar(x,y,color="green")
# plt.title("Binomial distribution")
# plt.show()


# Problem
# A disease affects 1% of a population
# a test is 95% accurate individuals and 90% accurate for non-diseased individuals
# find the probability of having disease given a postitive test result

def bayes_theorem(prior,sensitivity,specifycity):
    evidence = (sensitivity*prior)+ ((1- specifycity)*(1-prior))
    posterior = (sensitivity *prior)/ evidence
    return posterior

prior = 0.01
sensitivity = 0.95
specificity = 0.90
posterior = bayes_theorem(prior,sensitivity,specificity)

print(posterior)