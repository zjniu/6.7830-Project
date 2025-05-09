import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.nn as nn

from . import mmd, rbm, samplers, utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def rbm_relax_sample(args):
    
    args = utils.Args(args)

    utils.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
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

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

    else:
        viz = plot = None

    model.load_state_dict(torch.load(os.path.join(args.model_path, 'rbm.pt')))

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
    utils.makedirs("{}/gibbs".format(args.save_dir))
    for i in range(args.n_steps):
        if i % 10 == 0 and plot is not None:
            if args.data == "mnist":
                hx = samplers.threshold(new_samples)
            else:
                hx = new_samples
            plot("{}/gibbs/samples_gibbs_{}.png".format(args.save_dir, i), hx)
        if i % 10 == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gibbs'].append(log_stat)
            print("gibbs", i, stat, stat.log10())
        new_samples = model.gibbs_sample(new_samples, 1)


    r_model = samplers.BinaryRelaxedModel(args.n_visible, model)
    r_model.to(device)


    temps = [1.]
    for temp in temps:
        log_mmds['svgd'] = []
        target = lambda x: r_model.logp_surrogate(x, temp)
        x = model.init_dist.sample((args.n_test_samples,)).to(device)
        x = nn.Parameter(r_model.init_from_data(x))
        #x = nn.Parameter(r_model.base_dist.sample((args.n_test_samples, args.n_visible)).to(device))
        optim = torch.optim.Adam(params=[x], lr=args.lr)
        svgd = samplers.SVGD(optim)
        utils.makedirs("{}/svgd".format(args.save_dir))
        for i in range(args.n_steps):
            #svgd.step(x, target)
            svgd.discrete_step(x, r_model.logp_target, target)

            if i % 10 == 0 and plot is not None:
                if args.data == "mnist":
                    hx = samplers.threshold(x)
                else:
                    hx = x
                plot("{}/svgd/samples_svgd_{}_{}.png".format(args.save_dir, temp, i), hx)

            if i % 10 == 0:
                hard_samples = samplers.threshold(x)
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds['svgd'].append(log_stat)
                print("temp = {}, itr = {}, log-mmd = {:.4f}, ess = {:.4f}".format(temp, i, log_stat, svgd._ess))



    sampler = samplers.DiffSampler(args.n_visible, 1, fixed_proposal=False, approx=True, temp=2.)
    x = model.init_dist.sample((args.n_test_samples,)).to(device)

    log_mmds['gwg'] = []
    utils.makedirs("{}/gwg".format(args.save_dir))
    for i in range(args.n_steps):
        # do sampling and time it
        xhat = sampler.step(x.detach(), model).detach()

        # compute hamming dist
        cur_hops = (x != xhat).float().sum(-1).mean().item()

        # update trajectory
        x = xhat

        if i % 10 == 0 and plot is not None:
            plot("{}/gwg/samples_gwg_{}.png".format(args.save_dir, i), x)


        if i % 10 == 0:
            hard_samples = x
            stat = kmmd.compute_mmd(hard_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gwg'].append(log_stat)
            print("gwg, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(i, log_stat, cur_hops))


    temps = [2., 1., .5] if args.data == 'random' else [1.]
    for sampler in ["hmc", "mala"]:
        for temp in temps:
            for ss in [.1, .01, .001] if args.data == 'random' else [.001]:#[.001, .01, .1]:#, 1.]:
                name = "{}-{}-{}".format(sampler, temp, ss)
                log_mmds[name] = []

                log_temp = nn.Parameter(torch.tensor([temp]).log().to(device))
                # mala_samples = r_model.init(args.n_test_samples).to(device)
                x = model.init_dist.sample((args.n_test_samples,)).to(device)
                mala_samples = r_model.init_from_data(x)
                print("Burn in")
                utils.makedirs("{}/{}".format(args.save_dir, sampler))
                for i in range(args.n_steps):
                    if sampler == "hmc":
                        mala_samples, ar, _ = r_model.hmc_step(mala_samples, ss, 5, log_temp.exp().detach())
                        ar = ar.mean().item()
                    else:
                        mala_samples, ar = r_model.step(mala_samples, ss, log_temp.exp(), accept_dist="target", tt=args.tt)


                    if i % 10 == 0:
                        hard_samples = samplers.threshold(mala_samples)
                        stat = kmmd.compute_mmd(hard_samples, gt_samples)
                        print(sampler, temp, i, log_temp.mean().exp().item(), ss, ar, stat, stat.log10())
                        log_mmds[name].append(stat.log10().item())

                    if i % 10 == 0 and plot is not None:
                        hx = samplers.threshold(mala_samples)
                        plot("{}/{}/samples_{}_{}.png".format(args.save_dir, sampler, name, i), hx)


    plt.clf()
    for temp in log_mmds.keys():
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/results.png".format(args.save_dir))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        pickle.dump(log_mmds, f)

    plt.close()
