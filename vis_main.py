import math
import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib.colors import LogNorm
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline
import pandas as pd
from matplotlib import cm
import torch_optimizer as optim
import lr_schedulers as sched
from hyperopt import fmin, hp, tpe
from utils import *


def execute_steps(
    func, initial_state, optimizer_class, scheduler_class, cor, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config, momentum=args['momentum'])  # optimizer_config ==>> Search for the optimal learning rate 
    scheduler = scheduler_class(optimizer)
    steps = []
    lr_steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter+1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        scheduler.step()
        lr_steps.append(scheduler.get_last_lr()[0])
        steps[:, i] = x.detach().numpy()
    return steps, lr_steps


def objective_noisy_hill(params):
    lr = params['lr']
    optimizer_class = params['optimizer_class']
    scheduler_class = params['scheduler_class']
    cor = params['cor']
    initial_state = args['ini']
    minimum = args['minimal']
    optimizer_config = dict(lr=lr)
    num_iter = 200
    steps, lr_steps = execute_steps(
        function, initial_state, optimizer_class, scheduler_class, cor, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_noisy_hill_3D(track_all_steps):
    sns.set(style="dark")
    dirs = os.path.join('./docs', args['name'])
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    lr_recording = os.path.join(dirs, 'write_findlr.txt')
    with open(lr_recording, 'w') as f:
        f.close()
    x = np.linspace(*args['x_lins'])
    y = np.linspace(*args['y_lins'])
    minimum = args['minimal']
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y], lib=np)
    fig = plt.figure(figsize=(16, 16))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none', cmap=cm.coolwarm, linewidth=0.1, antialiased=True, shade=True,alpha=0.3)
    ax.contour(X,Y,Z,args['counter'],zdir='z',offset=args['offset'],extend="min", alpha=0.3, cmap=cm.coolwarm, antialiased=True)  # cm.coolwarm 'viridis'
    # ax.contour(X,Y,Z,levels=np.logspace(0,5,35),norm=LogNorm(),zdir='z',offset=args['offset'],extend="min", alpha=0.3, cmap=cm.coolwarm, antialiased=True)
    for grad_iter, optimizer_name, scheduler_name, cor, lr in track_all_steps:
        iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
        iter_z = function([iter_x, iter_y], lib=np)
        ax.plot(iter_x, iter_y, iter_z, color=cor, linestyle='-', linewidth=1., label=scheduler_name)
        plt.plot(iter_x[-1], iter_y[-1], function([iter_x[-1], iter_y[-1]], lib=np), color=cor, marker='D', markersize=6)
        print('LR:  {} | x:  {:.2f} | y:  {:.2f} | z:  {:.2f}'.format(scheduler_name, iter_x[-1], iter_y[-1], 
              function(torch.Tensor([iter_x[-1],iter_y[-1]])).item()))
        with open(lr_recording, 'a') as f:
            f.write(scheduler_name + ':  ' + str(lr) + '\n')
            f.close
    ax.set_title(
        '{}: {} with '
        '{} iterations'.format(args['name'], optimizer_name, len(iter_x))
    )
    ax.view_init(*args['view'])
    ax.set_xlabel('$x$',labelpad=0.1)
    ax.set_ylabel('$y$',labelpad=0.1)
    ax.set_zlabel('$z$',labelpad=0)
    ax.set_zlim(args['z_lim'])
    ax.set_xlim(args['x_lim'])
    ax.set_ylim(args['y_lim'])
    x_major_locator=MultipleLocator(args['x_major_locator'])
    x_minor_locator=MultipleLocator(args['x_minor_locator'])
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    y_major_locator=MultipleLocator(args['y_major_locator'])
    y_minor_locator=MultipleLocator(args['y_minor_locator'])
    ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    z_major_locator=MultipleLocator(args['z_major_locator'])
    z_minor_locator=MultipleLocator(args['z_minor_locator'])
    ax.zaxis.set_major_locator(z_major_locator)
    ax.zaxis.set_minor_locator(z_minor_locator)
    ax.tick_params(which='major',length=6,labelsize=10,direction='out',width=0.6,pad=0.4)
    ax.tick_params(which='minor',length=3,direction='out',width=0.3,pad=0.2)
    plt.plot(*minimum, function([*minimum], lib=np), 'rD', markersize=6, label='minimum point')
    plt.legend(loc='upper right', prop=config.font)
    plt.tick_params(axis='both',which='major',labelsize=10)
    if config.use_same_inilr:
        plt.savefig(os.path.join(dirs, '{}_3D_same_{}.tiff'.format(args['name'], optimizer_name)), dpi=600)
    else:
        plt.savefig(os.path.join(dirs, '{}_3D_diff_{}.tiff'.format(args['name'], optimizer_name)), dpi=600)

def plot_noisy_hill_2D(track_all_steps):
    sns.set(style="dark")
    dirs = os.path.join('./docs', args['name'])
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    x = np.linspace(*args['x_lins'])
    y = np.linspace(*args['y_lins'])
    minimum = args['minimal']
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y], lib=np)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, args['counter'], alpha=0.4, cmap=cm.coolwarm) 
    for grad_iter, optimizer_name, scheduler_name, cor, lr in track_all_steps:
        iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
        ax.plot(iter_x, iter_y, color=cor, linestyle='-', linewidth=0.8, label=scheduler_name)
        plt.plot(iter_x[-1], iter_y[-1], color=cor, marker='D', markersize=6)

    ax.set_title(
        '{}: {} with '
        '{} iterations'.format(args['name'], optimizer_name, len(iter_x))
    )
    ax.set_xlabel('$x$',labelpad=0.1)
    ax.set_ylabel('$y$',labelpad=0.1)
    ax.set_xlim(args['x_lim'])
    ax.set_ylim(args['y_lim'])
    x_major_locator=MultipleLocator(0.4)
    x_minor_locator=MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    y_major_locator=MultipleLocator(0.8)
    y_minor_locator=MultipleLocator(0.4)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.tick_params(which='major',length=6,labelsize=10,direction='out',width=0.6,pad=0.4)
    ax.tick_params(which='minor',length=3,direction='out',width=0.3,pad=0.2)
    plt.plot(*minimum, 'rD', markersize=6, label='minimum point')
    plt.legend(loc='upper right', prop=config.font)
    plt.tick_params(axis='both',which='major',labelsize=10)
    if config.use_same_inilr:
        plt.savefig(os.path.join(dirs, '{}_2D_same_{}.tiff'.format(args['name'], optimizer_name)), dpi=600)
    else:
        plt.savefig(os.path.join(dirs, '{}_2D_diff_{}.tiff'.format(args['name'], optimizer_name)), dpi=600)


def execute_experiments(
    optimizers, objective, func, plot_func3D, plot_func2D, initial_state, seed=1
):
    seed = seed
    lrs_steps = []
    all_steps = []
    track_all_steps = []
    schedulers_name = []
    colors = []
    inilr = {}
    for item in optimizers:
        optimizer_class, scheduler_class, cor, lr_low, lr_hi = item
        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'scheduler_class': hp.choice('scheduler_class', [scheduler_class]),
            'cor': hp.choice('cor', [cor]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,  # Specify the search algorithm(Random Search)
            max_evals=200,
            rstate=np.random.RandomState(seed),
        )
        number_iteration = config.number_iteration
        print('{}:  {:.2f} | {} | {}'.format(scheduler_class.__name__, best['lr'], optimizer_class, number_iteration+1))
        inilr[scheduler_class.__name__] = best['lr']
        if config.use_same_inilr:
            use_constant_inilr = inilr['constant_lr']
        else:
            use_constant_inilr = inilr[scheduler_class.__name__]
        steps, lr_steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            scheduler_class,
            cor,
            {'lr': use_constant_inilr},
            num_iter=number_iteration,
        )
        lrs_steps.append(lr_steps)
        all_steps.append(steps)
        schedulers_name.append(scheduler_class.__name__)
        colors.append(cor)
        track_all_steps.append([steps, optimizer_class.__name__, scheduler_class.__name__, cor, best['lr']])
    plot_func3D(track_all_steps)
    plot_func2D(track_all_steps)
    return all_steps, lrs_steps, schedulers_name, colors


def plot_lrs(lrs_steps, schedulers_name, colors, number_iteration=[config.number_iteration]):
    dirs = os.path.join('./docs', args['name'])
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if config.use_same_inilr:
        lr_recording = os.path.join(dirs, 'write_same_lr_{}.txt'.format(args['name']))
    else:
        lr_recording = os.path.join(dirs, 'write_diff_lr_{}.txt'.format(args['name']))
    with open(lr_recording, 'w') as f:
        f.close()
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 8))
    num = len(lrs_steps)
    num_iters = number_iteration*num
    min_max_lims = []
    for lr_steps, num_iter, scheduler_name, color in zip(lrs_steps, num_iters, schedulers_name, colors):
        num_iter = [i for i in range(1, num_iter+1)]
        min_max_lims.append(min(lr_steps))
        min_max_lims.append(max(lr_steps))
        lr_dict = {'x':num_iter, 'y':lr_steps}
        lr = pd.DataFrame(lr_dict)
        sns.lineplot(x="x", y="y", data=lr, color=color, label=scheduler_name)
        with open(lr_recording, 'a') as f:
            f.write(scheduler_name + '\n')
            for lr, itera in zip(lr_steps, num_iter):
                f.write('itera: {} || lr:  {:.4f} \n'.format(str(itera), lr))
            f.write('\n')
            f.close()
    plt.xlabel('iterations')
    plt.ylabel('LR')
    plt.xlim([0, number_iteration[0]])
    plt.ylim([min(min_max_lims), max(min_max_lims)+max(min_max_lims)/10])
    plt.legend(loc="upper right")
    plt.tick_params(axis='both',which='major',labelsize=10)
    if config.use_same_inilr:
        plt.savefig(os.path.join(dirs, 'lr_same_{}.tiff'.format(args['name'])), dpi=600, pad_inches=0)
    else:
        plt.savefig(os.path.join(dirs, 'lr_diff_{}.tiff'.format(args['name'])), dpi=600, pad_inches=0)

def plot_loss(all_steps, schedulers_name, colors, number_iteration=[config.number_iteration]):
    dirs = os.path.join('./docs', args['name'])
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if config.use_same_inilr:
        loss_recording = os.path.join(dirs, 'write_same_loss_{}.txt'.format(args['name']))
    else:
        loss_recording = os.path.join(dirs, 'write_diff_loss_{}.txt'.format(args['name']))
    with open(loss_recording, 'w') as f:
        f.close()
    minimum = args['minimal']
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 8))
    z_distances = []
    for step in all_steps:
        z_ = []
        for i in range(config.number_iteration):
            dis = np.sqrt((step[0][i]-minimum[0])**2 + (step[1][i]-minimum[1])**2)
            z_.append(dis)
        z_distances.append(z_)
    num = len(all_steps)
    num_iters = number_iteration*num
    for z, num_iter, scheduler_name, color in zip(z_distances, num_iters, schedulers_name, colors):
        num_iter = [i for i in range(1, num_iter+1)]
        x_new = np.linspace(min(num_iter), max(num_iter), 200)
        z_smooth = make_interp_spline(num_iter, z)(x_new)
        loss_dict = {'x':x_new, 'y':z_smooth}
        loss = pd.DataFrame(loss_dict)
        sns.lineplot(x="x", y="y", data=loss, color=color, label=scheduler_name)
        with open(loss_recording, 'a') as f:
            f.write(scheduler_name + '\n')
            for zz, itera in zip(z, num_iter):
                f.write('itera: {} || loss:  {:.4f} \n'.format(str(itera), zz))
            f.write('\n')
            f.close()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.xlim([0, number_iteration[0]])
    plt.legend(loc="upper right")
    plt.tick_params(axis='both',which='major',labelsize=10)
    if config.use_same_inilr:
        plt.savefig(os.path.join(dirs, 'loss_same_{}.tiff'.format(args['name'])), dpi=600, pad_inches=0)
    else:
        plt.savefig(os.path.join(dirs, 'loss_diff_{}.tiff'.format(args['name'])), dpi=600, pad_inches=0)

def LookaheadYogi(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)

if __name__ == '__main__':
    colors = config.colors
    # all vis functions <<rosenbrock, peaks, saddle, saddle2, cross_sectional, goldsteinprice, threehump, sigmoid>>
    function = squre
    args = config.make_math(function.__name__)
    optimizers = [
        # baselines
        (torch.optim.SGD, sched.constant_lr, colors[0], -8, -1.0),
        (torch.optim.SGD, sched.step_lr, colors[1], -8, -1.0),
        # (torch.optim.SGD, sched.cosine_lr, colors[2], -8, -1.0),
        # (torch.optim.SGD, sched.cosineclc_lr, colors[3], -8, -1.0),
        # (torch.optim.SGD, sched.lwarmcosine_lr, colors[4], -8, -1.0),
        # note change momentum opt
        # (optim.Yogi, sched.constant_lr, colors[0], -8, -0.1),
        # (optim.Yogi, sched.step_lr, colors[1], -8, -0.1),
        # (optim.Yogi, sched.cosine_lr, colors[2], -8, -0.1),
        # (optim.Yogi, sched.cosineclc_lr, colors[3], -8, -0.1),
        # (optim.Yogi, sched.lwarmcosine_lr, colors[4], -8, -0.1),
    ]

    
    all_steps, lrs_steps, schedulers_name, colors = execute_experiments(
        optimizers,
        objective_noisy_hill,
        function,
        plot_noisy_hill_3D,
        plot_noisy_hill_2D,
        args['ini']
    )
    plot_lrs(lrs_steps, schedulers_name, colors)
    plot_loss(all_steps, schedulers_name, colors)
