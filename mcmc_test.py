def mcmc(prior_dist, size=100000, burn=1000, thin=10, Z=3, N=10):
    import random
    from scipy.stats import binom
    #Make Markov chain (Monte Carlo)
    mc = [0] #Initialize markov chain
    while len(mc) < thin*size + burn:
        cand = random.gauss(mc[-1], 1) #Propose candidate
        ratio = (binom.pmf(Z, N, cand)*prior_dist(cand, size)) / (binom.pmf(Z, N, mc[-1])*prior_dist(mc[-1], size))
        if ratio > random.random(): #Acceptence criteria
        mc.append(cand)
    else:
        mc.append(mc[-1])
    #Take sample
    sample = []
    for i in range(len(mc)):
        if i >= burn and (i-burn)%thin == 0:
            sample.append(mc[i])
    sample = sorted(sample)
    #Estimate posterior probability
    post = []
    for p in sample:
        post.append(binom.pmf(Z, N, p) * prior_dist(p, size))
    return sample, post, mc

def uniform_prior_distribution(p, size):
    prior = 1.0/size
    return prior
