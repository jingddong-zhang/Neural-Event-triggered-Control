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

def normalize(data):
    return data/np.abs(data).max()

def process_data(case):
    hessv = np.load(osp.join('./data/fig4', case+'_Vxx_10.npy'))
    convexity = np.zeros(len(hessv))
    for i in range(len(hessv)):
        convex_list = []
        data = hessv[i]
        for j in range(len(data)):
            convex_list.append(np.real(np.linalg.eigvals(data[i,:,i,:]).mean()))
        convexity[i] = np.array(convex_list).mean()
    np.save(osp.join('./data/fig4', case+'_eigs_mean_10.npy'),convexity)
# process_data('PI')

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
    fontsize = 23
    ticksize = 18
    lw = 2
    MC_eigs = np.load('./data/fig4/MC_eigs_mean_10.npy')
    MC_ux = np.load('./data/fig4/MC_ux_10.npy')
    MC_ux = np.linalg.norm(MC_ux, axis=2).mean(axis=1)
    MC_times = np.load('./data/fig4/MC_times_10.npy')

    PI_eigs = np.load('./data/fig4/PI_eigs_mean_10.npy')[1:]
    PI_ux = np.load('./data/fig4/PI_ux_10.npy')
    PI_ux = np.linalg.norm(PI_ux, axis=2).mean(axis=1)[1:]
    PI_times = np.load('./data/fig4/PI_times_10.npy')[1:]
    # print(PI_eigs.shape,PI_ux.shape,PI_times.shape)
    PI_eigs = np.concatenate((PI_eigs,PI_eigs[-2:-1]))
    PI_ux = np.concatenate((PI_ux,PI_ux[-2:-1]))
    PI_times =  np.concatenate((PI_times,PI_times[-2:-1]))
    plt.figure(figsize=(9, 3.2))
    plt.subplots_adjust(left=0.04, bottom=0.12, right=0.97, top=0.88, hspace=0.2, wspace=0.20)
    plt.subplot(131)
    plot_grid()
    plt.plot(np.arange(len(PI_eigs)),PI_eigs,marker='o',color=colors[-1],label='NETC-PI')
    plt.plot(np.arange(len(MC_eigs)),MC_eigs,marker='o',color=colors[-2],label='NETC-MC')
    plt.xlabel('Epochs',fontsize=fontsize,labelpad=-15)
    plt.xticks([0,10],['0','10'],fontsize=ticksize)
    plt.ylabel('Convexity of V',fontsize=fontsize,labelpad=-10)
    plt.yticks([0, 8], ['0', '8'], fontsize=ticksize)
    plt.legend(fontsize=ticksize,loc=1,ncol=2,frameon=False,bbox_to_anchor=[2.5,1.17],labelcolor='k',handletextpad=0.5,borderpad=0.2,borderaxespad=0.3,handlelength=2.0)
    plt.text(0, 7, '(a)', fontsize=fontsize)

    plt.subplot(132)
    plot_grid()
    plt.plot(np.arange(len(PI_times)), PI_times, marker='o', color=colors[-1])
    plt.plot(np.arange(len(MC_times)), MC_times, marker='o', color=colors[-2])
    plt.xlabel('Epochs',fontsize=fontsize,labelpad=-15)
    plt.xticks([0,10],['0','10'],fontsize=ticksize)
    plt.ylabel('Triggering Times',fontsize=fontsize,labelpad=-15)
    plt.yticks([10, 70], ['10', '70'], fontsize=ticksize)
    plt.text(0.5, 63, '(b)', fontsize=fontsize)

    plt.subplot(133)
    plot_grid()
    plt.plot(np.arange(len(PI_ux)), PI_ux, marker='o', color=colors[-1])
    plt.plot(np.arange(len(MC_ux)), MC_ux, marker='o', color=colors[-2])
    plt.xlabel('Epochs',fontsize=fontsize,labelpad=-15)
    plt.xticks([0,10],['0','10'],fontsize=ticksize)
    plt.ylabel(r'$\|\nabla u\|$',fontsize=fontsize,labelpad=-15)
    plt.yticks([10, 25], ['10', '25'], fontsize=ticksize)
    plt.text(0, 23, '(c)', fontsize=fontsize)
    plt.show()

plot()