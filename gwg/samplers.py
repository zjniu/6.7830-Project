import torch
import torch.nn as nn
import torch.distributions as dists
from . import utils
import numpy as np


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []

        if self.fixed_proposal:
            delta = self.diff_fn(x, model)
            # probs = torch.softmax(delta, dim=-1)
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            # cd = dists.OneHotCategorical(probs=probs)
            cd = dists.OneHotCategorical(logits=delta)
            for i in range(self.n_steps):
                changes = cd.sample()

                x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                a_s.append(a.mean().item())
            self._ar = np.mean(a_s)
        else:
            for i in range(self.n_steps):
                forward_delta = self.diff_fn(x_cur, model)
                # forward_probs = torch.softmax(forward_delta, dim=-1)
                # forward_probs = forward_probs / forward_probs.sum(dim=-1, keepdim=True)
                # cd_forward = dists.OneHotCategorical(probs=forward_probs)
                cd_forward = dists.OneHotCategorical(logits=forward_delta)
                changes = cd_forward.sample()

                lp_forward = cd_forward.log_prob(changes)

                x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                reverse_delta = self.diff_fn(x_delta, model)
                # reverse_probs = torch.softmax(reverse_delta, dim=-1)
                # reverse_probs = reverse_probs / reverse_probs.sum(dim=-1, keepdim=True)
                # cd_reverse = dists.OneHotCategorical(probs=reverse_probs)
                cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                lp_reverse = cd_reverse.log_prob(changes)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # forward_probs = torch.softmax(forward_delta, dim=-1)
            # forward_probs = forward_probs / forward_probs.sum(dim=-1, keepdim=True)
            # cd_forward = dists.OneHotCategorical(probs=forward_probs)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            # reverse_probs = torch.softmax(reverse_delta, dim=-1)
            # reverse_probs = reverse_probs / reverse_probs.sum(dim=-1, keepdim=True)
            # cd_reverse = dists.OneHotCategorical(probs=reverse_probs)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class SVGD(nn.Module):
    def __init__(self, optim, kernel=RBF()):
        super().__init__()
        self.K = kernel
        self.optim = optim
        self._ess = 0.0

    def phi(self, X, log_prob):
        X = X.detach().requires_grad_(True)

        log_prob = log_prob(X)
        score_func = torch.autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def discrete_phi(self, X, log_prob_t, log_prob_s):
        X = X.detach().requires_grad_(True)

        log_prob_t = log_prob_t(X)
        log_prob_s = log_prob_s(X)
        log_w = log_prob_s - log_prob_t
        w = log_w.softmax(0).detach()
        score_func = torch.autograd.grad(log_prob_s.sum(), X)[0] * w[:, None]
        self._ess = 1./(w**2).sum()


        K_XX = self.K(X, X.detach())
        Kw = K_XX * w[None, :]
        grad_K1 = -torch.autograd.grad(Kw.sum(), X, create_graph=True)[0]

        # Kw = K_XX * w[:, None]
        # grad_K2 = -torch.autograd.grad(Kw.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K1) #/ X.size(0)
        return phi

    def step(self, X, log_prob):
        self.optim.zero_grad()
        X.grad = -self.phi(X, log_prob)
        self.optim.step()

    def discrete_step(self, X, log_prob_t, log_prob_s):
        self.optim.zero_grad()
        X.grad = -self.discrete_phi(X, log_prob_t, log_prob_s)
        self.optim.step()


def _ebm_helper(netEBM, x):
    x = x.clone().detach().requires_grad_(True)
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = x.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_ebm_samples(netEBM, x_init, burn_in, num_samples_posterior,
                    leapfrog_steps, stepsize,
                    flag_adapt=1, hmc_learning_rate=.02, hmc_opt_accept=.67, acceptEBM=None):
    if type(stepsize) != float:
        assert flag_adapt == 0
        stepsize = stepsize[:, None]
    device = x_init.device
    bsz, x_size = x_init.size(0), x_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, *x_size).to(device)
    current_x = x_init
    cnt = 0
    for i in range(n_steps):
        x = current_x
        p = torch.randn_like(current_x)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, current_x)
        current_U = -logjoint_vect.view(-1, 1)
        if acceptEBM is None:
            current_U_A = current_U
        else:
            logjoint_vect_A, _, _ = _ebm_helper(acceptEBM, current_x)
            current_U_A = -logjoint_vect_A.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            x = x + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
        proposed_U = -logjoint_vect.view(-1, 1)
        if acceptEBM is None:
            proposed_U_A = proposed_U
        else:
            logjoint_vect_A, _, _ = _ebm_helper(acceptEBM, x)
            proposed_U_A = -logjoint_vect_A.view(-1, 1)
        grad_U = -grad_logjoint

        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)       # should be size of B x 1
        proposed_K = 0.5 * (p**2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)     # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U_A - proposed_U_A + current_K - proposed_K))
        accept = accept.float().squeeze()       # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_x = current_x.clone()
            current_x[ind, :] = x[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        elif i >= burn_in:
            samples[cnt*bsz: (cnt+1)*bsz, :] = current_x
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def update_logp(u, u_mu, std):
    return dists.Normal(u_mu, std).log_prob(u).flatten(start_dim=1).sum(1)


def threshold(x):
    return (x > 0.).float()


def soft_threshold(x, t=1.):
    return (x / t).sigmoid()


class BinaryRelaxedModel(nn.Module):
    def __init__(self, dim, model, base="gaussian"):
        super().__init__()
        self.model = model
        if base == "gaussian":
            self.base_dist = dists.Normal(0., 1.)
        else:
            raise ValueError

        self.dim = dim

    def init(self, n):
        return self.base_dist.sample((n, self.dim))

    def init_from_data(self, x):
        x_c = self.base_dist.sample((x.size(0), self.dim)).to(x.device)
        x_c = x_c * torch.sign(x_c)  # make positive
        # [0, 1] --> [-1, 1]
        xp = 2 * x - 1
        x_out = xp * x_c
        return x_out

    def logp_target(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = threshold(x)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return base + m_cont

    def logp_surrogate(self, x, t=1.):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = soft_threshold(x, t=t)
        m_cont = self.model(x_proj).squeeze()  # .squeeze()
        return base + m_cont

    def logp_accept_obj(self, x, step_size, t):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x, t=t)
        grad = torch.autograd.grad(logp_s.sum(), x, retain_graph=True, create_graph=True)[0]
        step_std = (2 * step_size) ** .5

        update_mu = x + step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)

        logp_updates = self.logp_surrogate(update, t=t)
        reverse_grad = torch.autograd.grad(logp_updates.sum(), update, retain_graph=True, create_graph=True)[0]
        reverse_update_mu = update + step_size * reverse_grad

        logp_forward = update_logp(update, update_mu, step_std)
        logp_forward2 = update_logp(update.detach(), update_mu, step_std)
        logp_backward = update_logp(x, reverse_update_mu, step_std)

        logp_target_update = self.logp_target(update)
        logp_surrogate_update = self.logp_surrogate(update, t)
        rebar = (logp_target_update - logp_surrogate_update).detach() * logp_forward2 + logp_surrogate_update
        return rebar + logp_backward - logp_forward, (rebar,
                                                      logp_target_update, logp_surrogate_update,
                                                      logp_backward, logp_forward)

    def step(self, x, step_size, t=1., tt=1., accept_dist="target"):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x, t=t) * tt
        grad = torch.autograd.grad(logp_s.sum(), x)[0]
        step_std = (2 * step_size) ** .5

        update_mu = x + step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)

        logp_updates = self.logp_surrogate(update, t=t) * tt
        reverse_grad = torch.autograd.grad(logp_updates.sum(), update)[0]
        reverse_update_mu = update + step_size * reverse_grad

        logp_forward = update_logp(update, update_mu, step_std)
        logp_backward = update_logp(x, reverse_update_mu, step_std)
        if accept_dist == "surrogate":
            logp_accept = logp_updates + logp_backward - logp_s - logp_forward
        else:
            logp_t = self.logp_target(x) * tt
            logp_updates_t = self.logp_target(update) * tt
            logp_accept = logp_updates_t + logp_backward - logp_t - logp_forward

        p_accept = logp_accept.exp()
        accept = (torch.rand_like(p_accept) < p_accept).float()
        next_x = accept[:, None] * update + (1 - accept[:, None]) * x
        return next_x, accept.mean().item()

    def hmc_step(self, x, step_size, n_steps, t=1., accept_dist="target"):
        if accept_dist == "surrogate":
            x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=t).squeeze(),
                                               x, n_steps, 1, 5, step_size)
        else:
            x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=t).squeeze(),
                                               x.detach().requires_grad_(), n_steps, 1, 5, step_size,
                                               acceptEBM=lambda x: self.logp_target(x).squeeze())
        return x, ar, step_size

    def annealed_hmc(self, x, step_sizes, n_steps, max_lam):
        n_lam = len(step_sizes)
        lams = np.linspace(0., max_lam, n_lam + 1)[1:][::-1]
        n_steps_per_lam = n_steps // n_lam
        ars = []
        for i in range(len(step_sizes)):
            lam = lams[i]
            step_size = step_sizes[i]
            if i < len(step_sizes) - 1:
                next_lam = lams[i + 1]
                x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=lam).squeeze(),
                                                   x, n_steps_per_lam, 1, 5, step_size,
                                                   acceptEBM=lambda x: self.logp_surrogate(x, t=next_lam).squeeze())
            else:
                x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=lam).squeeze(),
                                                   x, n_steps_per_lam, 1, 5, step_size,
                                                   acceptEBM=lambda x: self.logp_target(x).squeeze())
            step_sizes[i] = step_size
            ars.append(ar.mean().item())
        return x, ars, step_sizes
