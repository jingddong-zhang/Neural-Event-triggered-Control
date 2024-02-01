import matplotlib.pyplot as plt
import numpy as np
import torch


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

def normalize(data):
    return data/np.abs(data).max()

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
    fig = plt.figure(figsize=(6.5, 3.1))
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.97, top=0.88, hspace=0.2, wspace=0.20)


    ax_0 = plt.subplot(121)
    PI = np.load('./data/sigma_PI.npy',allow_pickle=True).item()
    MC = np.load('./data/sigma_MC.npy',allow_pickle=True).item()
    PI_num = PI['num']
    MC_num = MC['num']
    PI_inter = PI['inter']
    MC_inter = MC['inter']
    plot_grid()
    k = 30

    def plot1(data, color, label,marker):
        mean, std = np.mean(data, axis=1), np.std(data, axis=1)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=color, alpha=0.2)
        plt.plot(np.arange(len(mean)), mean, color=color, label=label,marker=marker)
    plot1(PI_num,colors[-1],'NETC-PI','o')
    plot1(MC_num,colors[-2],'NETC-MC','o')
    plt.xticks([0, 8], ['0.1', '0.9'], fontsize=ticksize)
    plt.yticks([10, 90], fontsize=ticksize)
    plt.xlabel(r'$\sigma$',fontsize=fontsize,labelpad=-15)
    plt.ylabel('Triggering Times',fontsize=fontsize,labelpad=-15)
    plt.legend(fontsize=ticksize,loc=1,ncol=2,frameon=False,bbox_to_anchor=[1.75,1.15],labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=2.0)
    plt.text(0,75,'(a)',fontsize=fontsize)

    plt.subplot(122)
    plot_grid()
    plot1(PI_inter,colors[-1],'NETC-PI','o')
    plot1(MC_inter,colors[-2],'NETC-PI','o')
    plt.xticks([0, 8], ['0.1', '0.9'], fontsize=ticksize)
    plt.yticks([0, 0.1],['0','0.1'], fontsize=ticksize)
    plt.xlabel(r'$\sigma$',fontsize=fontsize,labelpad=-15)
    plt.ylabel('Inter-event Time',fontsize=fontsize,labelpad=-20)
    # plt.legend(fontsize=ticksize,loc=1,ncol=1,frameon=False,labelcolor='k',handletextpad=0.3,borderpad=0.2,borderaxespad=0.3,handlelength=1.0)
    plt.text(0,0.082,'(b)',fontsize=fontsize)

    # plt.subplot(236)
    # PI = np.load('./data/lambda_PI.npy',allow_pickle=True).item()
    # PI_num = PI['num']
    # PI_inter = PI['inter']
    # PI_mse = PI['mse']

    # plt.subplot(236)
    # MC = np.load('./data/lambda_MC.npy',allow_pickle=True).item()
    # MC_num = MC['num']
    # MC_inter = MC['inter']
    # MC_mse = MC['mse']
    # print(MC_num.shape)
    # box1 = plt.boxplot([MC_num[i] for i in range(1,4)],positions=[1,4,7],patch_artist=True,showmeans=True,
    #             boxprops={"facecolor": "C0",
    #                       "edgecolor": "grey",
    #                       "linewidth": 0.5},
    #             medianprops={"color": "k", "linewidth": 0.5},
    #             meanprops={'marker':'+',
    #                        'markerfacecolor':'k',
    #                        'markeredgecolor':'k',
    #                        'markersize':5})


    plt.show()

