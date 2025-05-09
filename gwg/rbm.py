import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow_probability as tfp
import torch
import torch.distributions as dists
import torch.nn as nn

from . import block_samplers, mmd, samplers, utils
from timeit import default_timer as timer
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BernoulliRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, data_mean=None):
        super().__init__()
        linear = nn.Linear(n_visible, n_hidden)
        self.W = nn.Parameter(linear.weight.data)
        self.b_h = nn.Parameter(torch.zeros(n_hidden,))
        self.b_v = nn.Parameter(torch.zeros(n_visible,))
        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((n_visible,)) * .5)
        self.data_dim = n_visible

    def p_v_given_h(self, h):
        logits = h @ self.W + self.b_v[None]
        return dists.Bernoulli(logits=logits)

    def p_h_given_v(self, v):
        logits = v @ self.W.t() + self.b_h[None]
        return dists.Bernoulli(logits=logits)

    def logp_v_unnorm(self, v):
        sp = torch.nn.Softplus()(v @ self.W.t() + self.b_h[None]).sum(-1)
        vt = (v * self.b_v[None]).sum(-1)
        return sp + vt

    def logp_v_unnorm_beta(self, v, beta):
        if len(beta.size()) > 0:
            beta = beta[:, None]
        vW = v @ self.W.t() * beta
        sp = torch.nn.Softplus()(vW + self.b_h[None]).sum(-1) - torch.nn.Softplus()(self.b_h[None]).sum(-1)
        #vt = (v * self.b_v[None]).sum(-1)
        ref_dist = torch.distributions.Bernoulli(logits=self.b_v)
        vt = ref_dist.log_prob(v).sum(-1)
        return sp + vt

    def forward(self, x):
        return self.logp_v_unnorm(x)

    def _gibbs_step(self, v):
        h = self.p_h_given_v(v).sample()
        v = self.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, n_steps=2000, n_samples=None, plot=False):
        if v is None:
            assert n_samples is not None
            v = self.init_dist.sample((n_samples,)).to(self.W.device)
        if plot:
           for i in tqdm(range(n_steps)):
               v = self._gibbs_step(v)
        else:
            for i in range(n_steps):
                v = self._gibbs_step(v)
        return v


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def rbm_train(args):

    args = utils.Args(args)
    
    utils.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)
    print(device)

    if args.data == "mnist":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device)
            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr,d.mean(), m.mean(),
                                                                                               (d - m).mean()))

    else:
        model.W.data = torch.randn_like(model.W.data) * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None

    torch.save(model.state_dict(), os.path.join(args.model_path, 'rbm.pt'))


def rbm_sample(args, temps):

    args = utils.Args(args)

    utils.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)
    print(device)

    if args.data == "mnist":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

    else:
        viz = plot = None

    model.load_state_dict(torch.load(os.path.join(args.model_path, 'rbm.pt')))

    ess_sample = dists.Bernoulli(probs=torch.ones((args.n_visible,)) * .5).sample((1,)).to(device)

    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    xs = {}
    chains = {}
    chain = []

    times['gibbs'] = []
    xs['gibbs'] = []
    start_time = timer()
    for i in range(args.n_steps):
        if i % args.print_every == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gibbs'].append(log_stat)
            print("gibbs", i, stat, stat.log10())
            times['gibbs'].append(timer() - start_time)
        new_samples = model.gibbs_sample(new_samples, 1)
        if i % args.save_sample_every == 0:
            xs['gibbs'].append(new_samples.cpu().numpy())
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(new_samples.cpu().numpy()[0][None])
            else:
                h = (new_samples != ess_sample).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])

    chain = np.concatenate(chain, 0)
    chains['gibbs'] = chain
    ess['gibbs'] = get_ess(chain, args.burn_in)
    print("ess = {} +/- {}".format(ess['gibbs'].mean(), ess['gibbs'].std()))

    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(args.n_visible, 1, fixed_proposal=False, approx=True, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(args.n_visible, 1, approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        xs[temp] = []
        chain = []
        cur_time = 0.
        for i in range(args.n_steps):
            # do sampling and time it
            st = timer()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += timer() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.save_sample_every == 0:
                xs['gibbs'].append(x.cpu().numpy())
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    h = (x != ess_sample).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            # if i % args.viz_every == 0 and plot is not None:
            #     plot("/{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(temp, i, log_stat, cur_hops))
        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

    ess_temps = temps
    plt.clf()
    plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps + ['gibbs']:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/results.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(ars[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/ars.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    for temp in temps:
        plt.clf()
        plt.plot(chains[temp][:, 0])
        plt.savefig("{}/trace_{}.png".format(args.save_dir, temp))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'ess': ess,
            'hops': hops,
            'log_mmds': log_mmds,
            'chains': chains,
            'xs': xs,
            'times': times
        }
        pickle.dump(results, f)

    plt.close()
