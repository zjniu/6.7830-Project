import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow_probability as tfp
import torch
import torch.distributions as dists
import torch.nn as nn
import igraph as ig

from . import block_samplers, samplers, utils
from timeit import default_timer as timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LatticeIsingModel(nn.Module):
    def __init__(self, dim, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False,
                 lattice_dim=2):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** lattice_dim,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = dim ** lattice_dim

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G * self.sigma

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b

def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_log_rmse(x):
    x = 2. * x - 1.
    x2 = (x ** 2).mean(-1)
    return x2.log10().detach().cpu().numpy()

def ising_sample(args, temps):

    args = utils.Args(args)

    utils.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    print(device)

    ess_sample = model.init_sample(1).to(device)

    hops = {}
    ess = {}
    times = {}
    xs = {}
    chains = {}
    means = {}

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(model.data_dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(model.data_dim, 1, fixed_proposal=False, approx=True, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(model.data_dim, 1, approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        times[temp] = []
        hops[temp] = []
        xs[temp] = []
        chain = []
        cur_time = 0.
        mean = torch.zeros_like(x)
        for i in range(args.n_steps):
            # do sampling and time it
            st = timer()
            xhat = sampler.step(x.detach(), model).detach()
            end = timer()
            cur_time += end - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            mean = mean + x
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x
                    h = (xc != ess_sample).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            times[temp].append(cur_time)
            hops[temp].append(cur_hops)
            if i % args.save_sample_every == 0:
                xs[temp].append(x.cpu().numpy())
            if i % args.print_every == 0:
                print("temp {}, itr = {}, hop-dist = {:.4f}".format(temp, i, cur_hops))

        means[temp] = mean / args.n_steps
        chain = np.concatenate(chain, 0)
        chains[temp] = chain
        if not args.no_ess:
            ess[temp] = get_ess(chain, args.burn_in)
            print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

    ess_temps = temps
    plt.clf()
    plt.boxplot([get_log_rmse(means[temp]) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/log_rmse.png".format(args.save_dir))

    if not args.no_ess:
        ess_temps = temps
        plt.clf()
        plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess.png".format(args.save_dir))

        plt.clf()
        plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    for temp in temps:
        plt.clf()
        plt.plot(chains[temp][:, 0])
        plt.savefig("{}/trace_{}.png".format(args.save_dir, temp))

    plt.close()

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'ess': ess,
            'hops': hops,
            'times': times,
            'xs': xs,
            'chains': chains,
            'means': means
        }
        pickle.dump(results, f)
