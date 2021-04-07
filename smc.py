from evaluator import evaluate
import torch
import numpy as np
import json
import sys



def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    log_weights = torch.tensor(log_weights)
    weights = torch.exp(log_weights)
    probs = weights / torch.sum(weights)
    indices = torch.multinomial(input=probs, num_samples=len(particles), replacement=True)

    new_particles = [particles[i] for i in indices]
    logZ = torch.logsumexp(log_weights, dim=0) + torch.log(torch.tensor(1/len(particles)))

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially

            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            elif res[2]['type'] == 'observe':
                cont, args, sigma = res
                weights[i] = sigma['log_prob']
                particles[i] = res
                address = particles[0][2]['alpha']
                assert address == sigma['alpha']

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)

        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in [1]:
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        log_Zs = []
        for count in [1, 10, 100, 1000, 10000, 100000]:
            n_particles = count
            logZ, particles = SMC(n_particles, exp)
            values = torch.stack(particles)
            log_Zs.append(logZ)
            print('program ', i, ' count ', count, ' Z: ', np.exp(logZ))
            print('program ', i, ' count ', count, ' mean: ', torch.mean(values.float(), dim=0))
            print('============================================')
            with open('program_' + str(i) + "_count_" + str(count) + ".npy", 'wb') as f:
                np.save(f, values)
            #TODO: some presentation of the results
        values = torch.stack(log_Zs)
        with open('program_' + str(i)+ "_logZs.npy", 'wb') as f:
            np.save(f, values)
