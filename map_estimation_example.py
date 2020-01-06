import csv

import pyro
import pyro.distributions as dist
import torch
import numpy as np
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

'''
Learn the MAP estimators of the log av and log var of the assumed lognormal distribution of average
avocado price measured over some years.
'''

def model():
    # Learn MAP of this, strarting from this prior distribution
    log_mean = pyro.sample('average_price_log_mean', dist.Gamma(torch.tensor(7.5), torch.tensor(1.)))
    log_var = pyro.sample('average_price_log_var', dist.Gamma(torch.tensor(7.5), torch.tensor(1.)))

    pyro.sample('average_price', dist.LogNormal(log_mean, log_var))

def guide():
    # only contains param and sample from Delta dist with
    av_price_log_mean_param = pyro.param('average_price_log_mean_param', torch.tensor(1.0))
    av_price_log_var_param = pyro.param('average_price_log_var_param', torch.tensor(1.0))
    return (
        pyro.sample('average_price_log_mean', dist.Delta(av_price_log_mean_param)),
        pyro.sample('average_price_log_var', dist.Delta(av_price_log_var_param)),
    )


def load_data():
    with open('avocado.csv') as avo:
        reader = csv.DictReader(avo)
        av_price = np.array([line['AveragePrice'] for line in reader]).astype(np.float32)
        data = torch.tensor(av_price)
    return data


data = load_data()
plt.hist(data.numpy())
plt.show()

conditioned_model = pyro.condition(model, data={"average_price": data})


optimizer = Adam({"lr": 0.001})

svi = pyro.infer.SVI(model=conditioned_model,
                     guide=guide,#
                     optim=optimizer,
                     loss=Trace_ELBO())

losses = []
log_means = []
log_vars = []
iters = 1000
for t in range(iters):
    losses.append(svi.step())
    log_mean = pyro.get_param_store().get_param('average_price_log_mean_param').item()
    log_var = pyro.get_param_store().get_param('average_price_log_var_param').item()
    log_means.append(log_mean)
    log_vars.append(log_var)
    if t%100 == 0:
        print('current loss: {} with log_mean={} and log_std={} at {} out of {} iterations'\
            .format(losses[-1], log_mean, log_var, t, iters))

        for k,v in pyro.get_param_store().items():
            print(k)
            print(v)


plt.plot(losses)
plt.show()


plt.subplot(211)
plt.plot(log_means)

plt.subplot(212)
plt.plot(log_vars)
plt.show()

plot_hist = data.numpy()
real_gamma_x = torch.linspace(0.5,3.5,1000)
real_gamma = torch.exp(
    dist.LogNormal(log_means[-1], log_vars[-1])\
        .log_prob(real_gamma_x)
    ).numpy()

plt.subplot(211)
plt.hist(plot_hist, 40, density=True)
plt.subplot(212)
plt.plot(real_gamma_x.numpy(),real_gamma)
plt.show()