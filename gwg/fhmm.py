import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn

from . import block_samplers, samplers, utils
from timeit import default_timer as timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FHMM(nn.Module):
    def __init__(self, N, K, W, W0, out_sigma, p, v,
                 learn_W=False, learn_W0=False, learn_p=False, learn_v=False, learn_obs=False):
        super().__init__()
        self.logit_v = nn.Parameter(v.log() - (1. - v).log(), requires_grad=learn_v)
        self.logit_p = nn.Parameter(p.log() - (1. - p).log(), requires_grad=learn_p)
        self.W = nn.Parameter(W, requires_grad=learn_W)
        self.W0 = nn.Parameter(W0, requires_grad=learn_W0)
        self.out_logsigma = nn.Parameter(torch.tensor(out_sigma).log(), requires_grad=learn_obs)
        self.K = K
        self.N = N

    @property
    def out_sigma(self):
        return self.out_logsigma.exp()

    def p_X0(self):
        return torch.distributions.Bernoulli(logits=self.logit_v)

    def p_XC(self):
        return torch.distributions.Bernoulli(logits=-self.logit_p)

    def log_p_X(self, X):
        X = X.view(X.size(0), self.N, self.K)
        X0 = X[:, 0]
        X_cur = X[:, :-1]
        X_next = X[:, 1:]
        X_change = (1. - X_cur) * X_next + (1. - X_next) * X_cur
        log_px0 = self.p_X0().log_prob(X0).sum(-1)
        log_pxC = self.p_XC().log_prob(X_change).sum(-1).sum(-1)
        return log_px0 + log_pxC

    def p_y_given_x(self, X, sigma=None):
        X = X.view(X.size(0), self.N, self.K)
        xw = (self.W[None, None] * X).sum(-1)
        mu = xw + self.W0
        if sigma is None:
            sigma = self.out_sigma
        out_dist = torch.distributions.Normal(mu, sigma)
        return out_dist

    def log_p_y_given_x(self, y, X, sigma=None):
        out_dist = self.p_y_given_x(X, sigma=sigma)
        if len(y.size()) == 1:
            return out_dist.log_prob(y[None]).sum(-1)
        else:
            return out_dist.log_prob(y).sum(-1)

    def log_p_joint(self, y, X, sigma=None):
        logp_y = self.log_p_y_given_x(y, X, sigma=sigma)
        logp_X = self.log_p_X(X)
        return logp_y + logp_X

    def sample_X(self, n=1):
        X0 = self.p_X0().sample((n,))
        XNs = [X0[:, None, :]]
        for i in range(self.N - 1):
            XC = self.p_XC().sample((n,))[:, None, :]
            X_cur = XNs[-1]
            X_next = (1. - XC) * X_cur + XC * (1. - X_cur)
            XNs.append(X_next)
        return torch.cat(XNs, 1)

def fhmm_sample(args, temps):

    args = utils.Args(args)

    utils.makedirs("{}/sources".format(args.save_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    W = args.W_init_sigma * torch.randn((args.K,))
    W0 = args.W_init_sigma * torch.randn((1,))
    p = args.X_keep_prob * torch.ones((args.K,))
    v = args.X0_mean * torch.ones((args.K,))

    model = FHMM(args.N, args.K, W, W0, args.obs_sigma, p, v)
    model.to(device)
    print("device is", device)

    # generate data
    Xgt = model.sample_X(1)
    p_y_given_Xgt = model.p_y_given_x(Xgt)

    mu = p_y_given_Xgt.loc
    mu_true = mu[0]
    plt.clf()
    plt.plot(mu_true.detach().cpu().numpy(), label="mean")
    ygt = p_y_given_Xgt.sample()[0]
    plt.plot(ygt.detach().cpu().numpy(), label='sample')
    plt.legend()
    plt.savefig("{}/data.png".format(args.save_dir))
    ygt = ygt.to(device)

    for k in range(args.K):
        plt.clf()
        plt.plot(Xgt[0, :, k].detach().cpu().numpy())
        plt.savefig("{}/sources/x_{}.png".format(args.save_dir, k))


    logp_joint_real = model.log_p_joint(ygt, Xgt).item()
    print("joint likelihood of real data is {}".format(logp_joint_real))

    log_joints = {}
    diffs = {}
    times = {}
    recons = {}
    ars = {}
    hops = {}
    phops = {}
    mus = {}

    dim = args.K * args.N
    x_init = model.sample_X(args.n_test_samples).to(device)
    samp_model = lambda _x: model.log_p_joint(ygt, _x)

    for temp in temps:
        utils.makedirs("{}/{}".format(args.save_dir, temp))
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(dim, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(dim, 1, fixed_proposal=False, approx=True, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(dim, 1, approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")
        
        x = x_init.clone().view(x_init.size(0), -1)

        diffs[temp] = []

        log_joints[temp] = []
        ars[temp] = []
        hops[temp] = []
        phops[temp] = []
        recons[temp] = []
        start_time = timer()
        for i in range(args.n_steps + 1):
            sm = samp_model
            xhat = sampler.step(x.detach(), sm).detach()

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()
            # update trajectory
            x = xhat

            if i % 1000 == 0:
                p_y_given_x = model.p_y_given_x(x)
                mu = p_y_given_x.loc
                plt.clf()
                plt.plot(mu_true.detach().cpu().numpy(), label="true")
                plt.plot(mu[0].detach().cpu().numpy() + .01, label='mu0')
                plt.plot(mu[1].detach().cpu().numpy() - .01, label='mu1')
                plt.legend()
                plt.savefig("{}/{}/mean_{}.png".format(args.save_dir, temp, i))
                mus[temp] = mu[0].detach().cpu().numpy()

            if i % 10 == 0:
                p_y_given_x = model.p_y_given_x(x)
                mu = p_y_given_x.loc
                err = ((mu - ygt[None]) ** 2).sum(1).mean()
                recons[temp].append(err.item())

                log_j = model.log_p_joint(ygt, x)
                diff = (x.view(x.size(0), args.N, args.K) != Xgt).float().view(x.size(0), -1).mean(1)
                log_joints[temp].append(log_j.mean().item())
                diffs[temp].append(diff.mean().item())
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, log-joint = {:.4f}, "
                      "hop-dist = {:.4f}, recons = {:.4f}".format(temp, i, log_j.mean().item(), cur_hops, err.item()))

        for k in range(args.K):
            plt.clf()
            xr = x.view(x.size(0), args.N, args.K)
            plt.plot(xr[0, :, k].detach().cpu().numpy())
            plt.savefig("{}/{}/source_{}.png".format(args.save_dir, temp, k))

        times[temp] = timer() - start_time


    plt.clf()
    for temp in temps:
        plt.plot(log_joints[temp], label=temp)
    plt.plot([logp_joint_real for _ in log_joints[temp]], label="true")
    plt.legend()
    plt.savefig("{}/joints.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(recons[temp], label=temp)
    plt.legend()
    plt.savefig("{}/recons.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(diffs[temp], label=temp)
    plt.legend()
    plt.savefig("{}/errs.png".format(args.save_dir))

    plt.clf()
    for i, temp in enumerate(temps):
        plt.plot(mus[temp] + float(i) * .01, label=temp)
    plt.plot(mu_true.detach().cpu().numpy(), label="true")
    plt.legend()
    plt.savefig("{}/mean.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'hops': hops,
            'recons': recons,
            'joints': log_joints,
            'times': times,
            'logp_joint_real': logp_joint_real
        }
        pickle.dump(results, f)

    plt.close()
