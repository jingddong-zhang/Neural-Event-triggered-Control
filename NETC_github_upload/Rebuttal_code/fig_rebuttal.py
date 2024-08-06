import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [31/255,145/255,158/255],
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
    [153/255,193/255,218/255], # sky blue
    [249/255,128/255,124/255], # rose red
    [112/255,48/255,160/255], #purple
    [255 / 255, 192 / 255, 0 / 255],  # gold

]
colors = np.array(colors)
def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.,zorder=0)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1,zorder=0)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

def plot():
    import matplotlib
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    matplotlib.rcParams['text.usetex'] = True

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


    size = 30
    fontsize = 18
    ticksize = 15
    lw = 2

    plt.figure(figsize=(8,10))
    plt.subplots_adjust(left=0.07, bottom=0.05, right=0.97, top=0.95, hspace=0.2, wspace=0.20)
    plt.rc('font', family='Times New Roman')

    ax1 = plt.subplot(321)
    plot_grid()
    netc = np.load('./rebuttal/NETC_high.npy',allow_pickle=True).item()
    ca = np.load('./rebuttal/Critic_Actor.npy',allow_pickle=True).item()
    etc = np.load('./rebuttal/Classic_ETC.npy',allow_pickle=True).item()
    netc_x,ca_x,etc_x = np.zeros([5,1000]),np.zeros([5,1000]),np.zeros([5,1000])
    netc_num,ca_num,etc_num = 0,0,0
    for i in range(5):
        netc_x[i] = np.sqrt(netc[f'{i}']['x']**2+netc[f'{i}']['y']**2+netc[f'{i}']['z']**2)
        ca_x[i] = np.sqrt(ca[f'{i}']['x']**2+ca[f'{i}']['y']**2+ca[f'{i}']['z']**2)
        etc_x[i] = np.sqrt(etc[f'{i}']['x']**2+etc[f'{i}']['y']**2+etc[f'{i}']['z']**2)
        netc_num += netc[f'{i}']['num']
        ca_num += ca[f'{i}']['num']
        etc_num += etc[f'{i}']['num']
    netc_num = int(netc_num/5)
    ca_num = int(ca_num / 5)
    etc_num = int(etc_num / 5)
    t = np.linspace(0,4,1000)
    netc_mean,ca_mean,etc_mean = netc_x.mean(axis=0),ca_x.mean(axis=0),etc_x.mean(axis=0)
    netc_std, ca_std, etc_std = netc_x.std(axis=0), ca_x.std(axis=0), etc_x.std(axis=0)
    plt.plot(t,netc_mean,c=colors[0],label=f'NETC-MC: {netc_num}',zorder=0)
    plt.fill_between(t,netc_mean-netc_std,netc_mean+netc_std,color=colors[0],alpha=0.5,zorder=0)
    plt.plot(t,ca_mean,c=colors[1],label=f'Critic-Actor: {ca_num}',zorder=1)
    plt.fill_between(t,ca_mean-ca_std,ca_mean+ca_std,color=colors[1],alpha=0.5,zorder=1)
    plt.plot(t,etc_mean,c=colors[2],label=f'Classic ETC: {etc_num}',zorder=1)
    plt.fill_between(t,etc_mean-etc_std,etc_mean+etc_std,color=colors[2],alpha=0.5,zorder=1)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-10)
    plt.xticks([0,4],['0','4'],fontsize=ticksize)
    plt.ylabel('MSE',fontsize=fontsize,labelpad=-15)
    plt.yticks([0, 10], ['0', '10'], fontsize=ticksize)
    plt.legend(fontsize=ticksize,loc=1,ncol=3,frameon=False,labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=2.0,bbox_to_anchor=[2.15,1.15],columnspacing=2.0)
    plt.text(0.1, 10.5, '(a)', fontsize=fontsize)

    axins = inset_axes(ax1,  2.2, 1.3, loc=2, bbox_to_anchor=(0.15, 0.95),
                       bbox_transform=ax1.figure.transFigure)

    L = 250
    tt = t[-L:]
    axins.plot(tt, netc_mean[-L:], c=colors[0], label='NETC-MC', zorder=0)
    axins.fill_between(tt, netc_mean[-L:] - netc_std[-L:], netc_mean[-L:] + netc_std[-L:], color=colors[0], alpha=0.5, zorder=0)
    axins.plot(tt, ca_mean[-L:], c=colors[1], label='Critic-Actor', zorder=1)
    axins.fill_between(tt, ca_mean[-L:] - ca_std[-L:], ca_mean[-L:] + ca_std[-L:], color=colors[1], alpha=0.5, zorder=1)
    axins.plot(tt, etc_mean[-L:], c=colors[2], label='Classic ETC', zorder=1)
    axins.fill_between(tt, etc_mean[-L:] - etc_std[-L:], etc_mean[-L:] + etc_std[-L:], color=colors[2], alpha=0.5, zorder=1)
    plt.xticks([])
    plt.yticks([])
    # axins.set_xticks([3, 4],['3','4'])
    # axins.set_yticks([0, 0.5])
    # xyA = (1.0-0.07, 5.3)
    # xyB = (3.0, 0.0)
    xyA = (1.0-0.3, 5.8)
    xyB = (3.0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con1 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='k')
    ax1.add_artist(con1)
    xyA = (4.0-0.30, 6.0)
    xyB = (4.0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con2 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='k')
    ax1.add_artist(con2)

    plt.subplot(322)
    plot_grid()
    netc_x, ca_x, etc_x = np.zeros([5, 21]), np.zeros([5, 21]), np.zeros([5, 21])
    for i in range(5):
        netc_x[i] = np.linalg.norm(netc[f'{i}']['traj_events'],ord=2,axis=1)[:21]
        ca_x[i] = np.linalg.norm(ca[f'{i}']['traj_events'],ord=2,axis=1)[:21]
        etc_x[i] = np.linalg.norm(etc[f'{i}']['traj_events'],ord=2,axis=1)[:21]
    triggers = np.linspace(0,20,21)
    netc_err = np.concatenate((netc_x.std(axis=0).reshape(1,-1),netc_x.std(axis=0).reshape(1,-1)),axis=0)
    ca_err = np.concatenate((ca_x.std(axis=0).reshape(1,-1),ca_x.std(axis=0).reshape(1,-1)),axis=0)
    etc_err = np.concatenate((etc_x.std(axis=0).reshape(1,-1),etc_x.std(axis=0).reshape(1,-1)),axis=0)

    plt.errorbar(triggers, netc_x.mean(axis=0), yerr=netc_err,color = colors[0],errorevery=2,fmt='-',elinewidth=2, capsize=2, capthick=1,)
    plt.errorbar(triggers, ca_x.mean(axis=0), yerr=ca_err, color=colors[1],errorevery=2,fmt='-',elinewidth=2, capsize=2, capthick=1)
    plt.errorbar(triggers, etc_x.mean(axis=0), yerr=etc_err, color=colors[2],errorevery=2,fmt='-',elinewidth=2, capsize=2, capthick=1)
    plt.xlabel('Triggering times', fontsize=fontsize, labelpad=-10)
    plt.xticks([0, 20], ['0', '20'], fontsize=ticksize)
    plt.ylabel('MSE', fontsize=fontsize, labelpad=-15)
    plt.yticks([0, 10], ['0', '10'], fontsize=ticksize)
    plt.text(17.1, 10.5, '(b)', fontsize=fontsize)


    plt.subplot(323)
    plot_grid()
    data = np.load('./rebuttal/parameter_noise.npy', allow_pickle=True).item()
    sigma_traj,rho_traj,beta_traj = np.zeros([5,1000]),np.zeros([5,1000]),np.zeros([5,1000])
    for i in range(5):
        sub_sigma_traj, sub_rho_traj, sub_beta_traj = np.zeros([5, 1000]), np.zeros([5, 1000]), np.zeros([5, 1000])
        for j in range(5):
            # sub_sigma_traj[j] = (np.linalg.norm(data[f'sigma_{i}'][f'{j}']['traj'],ord=2,axis=1))
            # sub_rho_traj[j] = (np.linalg.norm(data[f'rho_{i}'][f'{j}']['traj'],ord=2,axis=1))
            # sub_beta_traj[j] = (np.linalg.norm(data[f'beta_{i}'][f'{j}']['traj'],ord=2,axis=1))
            sub_sigma_traj[j] = np.log(np.linalg.norm(data[f'sigma_{i}'][f'{j}']['traj'], ord=2, axis=1))
            sub_rho_traj[j] = np.log(np.linalg.norm(data[f'rho_{i}'][f'{j}']['traj'], ord=2, axis=1))
            sub_beta_traj[j] = np.log(np.linalg.norm(data[f'beta_{i}'][f'{j}']['traj'], ord=2, axis=1))
        sigma_traj[i] = sub_sigma_traj.mean(axis=0)
        rho_traj[i] = sub_rho_traj.mean(axis=0)
        beta_traj[i] = sub_beta_traj.mean(axis=0)
        # sigma_traj[i] = np.linalg.norm(data[f'sigma_{i}'][f'{0}']['traj'],ord=2,axis=1)
        # rho_traj[i] = np.linalg.norm(data[f'rho_{i}'][f'{0}']['traj'],ord=2,axis=1)
        # beta_traj[i] = np.linalg.norm(data[f'beta_{i}'][f'{0}']['traj'],ord=2,axis=1)

    sigma_mean,rho_mean,beta_mean = sigma_traj.mean(axis=0),rho_traj.mean(axis=0),beta_traj.mean(axis=0)
    sigma_std, rho_std, beta_std = sigma_traj.std(axis=0), rho_traj.std(axis=0), beta_traj.std(axis=0)
    plt.plot(t,sigma_mean,c=colors[0],label=r'$\sigma\sim N(0,1)$',zorder=0)
    plt.fill_between(t,sigma_mean-sigma_std,sigma_mean+sigma_std,color=colors[0],alpha=0.5,zorder=0)
    plt.plot(t,rho_mean,c=colors[1],label=r'$\rho\sim N(0,1)$',zorder=1)
    plt.fill_between(t,rho_mean-rho_std,rho_mean+rho_std,color=colors[1],alpha=0.5,zorder=1)
    plt.plot(t,beta_mean,c=colors[2],label=r'$\beta\sim N(0,1)$',zorder=1)
    plt.fill_between(t,beta_mean-beta_std,beta_mean+beta_std,color=colors[2],alpha=0.5,zorder=1)
    plt.legend(fontsize=ticksize,loc=1,ncol=1,frameon=True,labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=1.0,columnspacing=0.5)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-10)
    plt.xticks([0,4],['0','4'],fontsize=ticksize)
    # plt.ylabel('MSE',fontsize=fontsize,labelpad=-15)
    # plt.yticks([0, 10], ['0', '10'], fontsize=ticksize)
    plt.yticks([-4, 0,2], ['-4','0', '2'], fontsize=ticksize)
    plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    plt.text(0.1, 1.5, '(c)', fontsize=fontsize)

    # plt.subplot(323)
    # plot_grid()
    # data = np.load('./rebuttal/parameter_noise_v1.npy', allow_pickle=True).item()
    # low, middle, high = np.zeros([5, 1000]), np.zeros([5, 1000]), np.zeros([5, 1000])
    # for i in range(5):
    #     low[i] = (np.linalg.norm(data[f'intensity_{0.5}'][f'{i}']['traj'], ord=2, axis=1))
    #     middle[i] = (np.linalg.norm(data[f'intensity_{1.0}'][f'{i}']['traj'], ord=2, axis=1))
    #     high[i] = (np.linalg.norm(data[f'intensity_{2.0}'][f'{i}']['traj'], ord=2, axis=1))
    #
    # low_mean, middle_mean, high_mean = low.mean(axis=0), middle.mean(axis=0), high.mean(axis=0)
    # low_std, middle_std, high_std = low.std(axis=0), middle.std(axis=0), high.std(axis=0)
    # plt.plot(t, low_mean, c=colors[0], label=r'$D_{x,y,z}=0.1$', zorder=0)
    # plt.fill_between(t, low_mean - low_std, low_mean + low_std, color=colors[0], alpha=0.5, zorder=0)
    # plt.plot(t, middle_mean, c=colors[1], label=r'$D_{x,y,z}=0.3$', zorder=0)
    # plt.fill_between(t, middle_mean - middle_std, middle_mean + middle_std, color=colors[1], alpha=0.5, zorder=0)
    # plt.plot(t, middle_mean, c=colors[2], label=r'$D_{x,y,z}=0.5$', zorder=0)
    # plt.fill_between(t, middle_mean - middle_std, middle_mean + middle_std, color=colors[2], alpha=0.5, zorder=0)
    # plt.legend(fontsize=ticksize, loc=1, ncol=1, frameon=True, labelcolor='k', handletextpad=0.5, borderpad=0.2,
    #            borderaxespad=0.3, handlelength=1.0, columnspacing=0.5)
    # plt.legend(fontsize=ticksize, loc=1, ncol=1, frameon=True, labelcolor='k', handletextpad=0.5, borderpad=0.2,
    #            borderaxespad=0.3, handlelength=1.0, columnspacing=0.5)
    # plt.xlabel('Time', fontsize=fontsize, labelpad=-10)
    # plt.xticks([0, 4], ['0', '4'], fontsize=ticksize)
    # plt.ylabel('MSE', fontsize=fontsize, labelpad=-15)
    # plt.yticks([0, 10], ['0', '10'], fontsize=ticksize)
    # # plt.yticks([-4, 0,2], ['-4','0', '2'], fontsize=ticksize)
    # # plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    # plt.text(0.1, 1.5, '(c)', fontsize=fontsize)

    plt.subplot(324)
    plot_grid()
    data = np.load('./rebuttal/noise_process.npy', allow_pickle=True).item()
    intensity_list = [0.1, 0.3, 0.5]
    low,middle,high = np.zeros([5,1000]),np.zeros([5,1000]),np.zeros([5,1000])
    for i in range(5):
        low[i] = np.log(np.linalg.norm(data[f'noise_{0.1}'][f'{i}']['traj'],ord=2,axis=1))
        middle[i] = np.log(np.linalg.norm(data[f'noise_{0.3}'][f'{i}']['traj'], ord=2, axis=1))
        high[i] = np.log(np.linalg.norm(data[f'noise_{0.5}'][f'{i}']['traj'], ord=2, axis=1))
        # low[i] = (np.linalg.norm(data[f'noise_{0.1}'][f'{i}']['traj'],ord=2,axis=1))
        # middle[i] = (np.linalg.norm(data[f'noise_{0.3}'][f'{i}']['traj'], ord=2, axis=1))
        # high[i] = (np.linalg.norm(data[f'noise_{0.5}'][f'{i}']['traj'], ord=2, axis=1))
    low_mean, middle_mean, high_mean = low.mean(axis=0), middle.mean(axis=0), high.mean(axis=0)
    low_std, middle_std, high_std = low.std(axis=0), middle.std(axis=0), high.std(axis=0)
    plt.plot(t, low_mean, c=colors[0], label=r'$D_{x,y,z}=0.1$', zorder=0)
    plt.fill_between(t, low_mean - low_std, low_mean + low_std, color=colors[0], alpha=0.5, zorder=0)
    plt.plot(t, middle_mean, c=colors[1], label=r'$D_{x,y,z}=0.3$', zorder=0)
    plt.fill_between(t, middle_mean - middle_std, middle_mean + middle_std, color=colors[1], alpha=0.5, zorder=0)
    plt.plot(t, middle_mean, c=colors[2], label=r'$D_{x,y,z}=0.5$', zorder=0)
    plt.fill_between(t, middle_mean - middle_std, middle_mean + middle_std, color=colors[2], alpha=0.5, zorder=0)
    plt.legend(fontsize=ticksize,loc=1,ncol=1,frameon=True,labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=1.0,columnspacing=0.5)
    plt.xlabel('Time', fontsize=fontsize, labelpad=-10)
    plt.xticks([0, 4], ['0', '4'], fontsize=ticksize)
    plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    plt.yticks([-4, 0,2], ['-4','0', '2'], fontsize=ticksize)
    # plt.ylabel('MSE',fontsize=fontsize,labelpad=-15)
    # plt.yticks([0, 10], ['0', '10'], fontsize=ticksize)
    plt.text(0.1, 1.5, '(d)', fontsize=fontsize)

    plt.subplot(325)
    plot_grid()
    data = np.load('./rebuttal/NETC_MC_Loss.npy', allow_pickle=True).item()
    lambda_list = [0.1, 1.0]
    lr_list = [0.01,0.05,0.1]
    epochs = np.arange(550)
    i = 0
    for lambda_1 in lambda_list:
        for lr in lr_list:
            loss = np.log(data[f'lambda_1={lambda_1} lr={lr}']) # size: (5,550)
            err = np.concatenate((loss.std(axis=0).reshape(1, -1), loss.std(axis=0).reshape(1, -1)), axis=0)
            print(loss.shape,err.shape)
            plt.errorbar(epochs, loss.mean(axis=0), yerr=err, color=colors[i], errorevery=50, fmt='-',
                         elinewidth=2,
                         capsize=2, capthick=1, label=r'$\lambda_2={}$,lr={}'.format(lambda_1,lr))
            i += 1
    plt.legend(fontsize=ticksize-3,loc=1,ncol=2,frameon=True,labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=1.0,columnspacing=0.2)
    plt.xlabel('Epochs', fontsize=fontsize, labelpad=-10)
    plt.xticks([0, 550], ['0', '550'], fontsize=ticksize)
    # plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    # plt.yticks([-4, 0,2], ['-4','0', '2'], fontsize=ticksize)
    plt.ylabel(r'$\log_{10}(\text{loss})$', fontsize=fontsize, labelpad=-5)
    plt.yticks([-4, 0,4], ['-4', '0','4'], fontsize=ticksize)
    plt.text(1.0, 1.0, '(e)', fontsize=fontsize)

    plt.subplot(326)
    plot_grid()
    data = np.load('./rebuttal/NETC_PI_Loss.npy', allow_pickle=True).item()
    lambda_list = [0.005,0.05]
    lr_list = [0.01,0.05,0.1]
    epochs = np.arange(100-1)
    i = 0
    for lambda_1 in lambda_list:
        for lr in lr_list:
            loss = np.log(data[f'lambda_1={lambda_1} lr={lr}'])[:,-100+1:]  # size: (5,600)
            err = np.concatenate((loss.std(axis=0).reshape(1, -1), loss.std(axis=0).reshape(1, -1)), axis=0)
            plt.errorbar(epochs, loss.mean(axis=0), yerr=err, color=colors[i], errorevery=10, fmt='-',
                         elinewidth=2,
                         capsize=2, capthick=1, label=r'$\lambda_2={}$,lr={}'.format(lambda_1, lr))
            i += 1
    plt.legend(fontsize=ticksize-3, loc=1, ncol=2, frameon=True, labelcolor='k', handletextpad=0.5, borderpad=0.2,
               borderaxespad=0.3, handlelength=1.0, columnspacing=0.2)
    plt.xlabel('Epochs', fontsize=fontsize, labelpad=-10)
    plt.xticks([0, 100], ['0', '100'], fontsize=ticksize)
    # plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    # plt.yticks([-4, 0,2], ['-4','0', '2'], fontsize=ticksize)
    plt.ylabel(r'$\log_{10}(\text{loss})$', fontsize=fontsize, labelpad=-5)
    plt.yticks([-4, 0, 4], ['-4', '0', '4'], fontsize=ticksize)
    plt.text(1.0, .5, '(f)', fontsize=fontsize)

    plt.savefig('./rebuttal/fig_rebuttal_v3.pdf')
    # plt.show()

plot()
