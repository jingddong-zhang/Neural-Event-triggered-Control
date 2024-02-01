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

nlc =  torch.load('nlc_event times_2000.pt')
nlc = np.array([_.detach().numpy() for _ in nlc])
nlc = nlc[np.where(nlc<=1.0)[0]]

netc =  torch.load('netc_event times.pt')
netc = np.array([_.detach().numpy() for _ in netc])
netc = netc[np.where(netc<=1.0)[0]]


t = np.linspace(0,2,len(nlc))

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


# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Times New Roman']})
# rc('font',**{'family':'serif','serif':['Times New Roman']})
# rc('text',usetex=True)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

size = 30
fontsize = 23
ticksize = 18
lw = 2
fig = plt.figure(figsize=(9, 3))
plt.subplots_adjust(left=0.07, bottom=0.12, right=0.97, top=0.88, hspace=0.2, wspace=0.17)

ax00 = plt.subplot(132)
nlc_cont = np.load('nlc_control.npy')[:len(nlc)+1,0]
netc_cont = np.load('netc_control.npy')[:len(netc)+1,0]
nlc_cont = normalize(nlc_cont)
netc_cont = normalize(netc_cont)
plot_grid()
k = 30
plt.plot(np.linspace(0,netc[0],k),netc_cont[0]*np.ones(k),color=colors[-1],label='NETC',lw = lw)
for i in range(len(netc)-1):
    plt.plot(np.linspace(netc[i],netc[i+1],k),netc_cont[i+1]*np.ones(k),c=colors[-1],lw=lw)

# ax01 = plt.subplot(131)
plt.plot(np.linspace(0,nlc[0],k),nlc_cont[0]*np.ones(k),color=colors[-2],label='NLC',lw=lw)
for i in range(len(nlc)-1):
    plt.plot(np.linspace(nlc[i],nlc[i+1],k),nlc_cont[i+1]*np.ones(k),c=colors[-2],lw=lw)
plt.yticks([-1,1],fontsize=ticksize)
plt.xticks([0,nlc[-1]],['0','1'],fontsize=ticksize)
plt.xlabel(r'$\text{Time}$',fontsize=fontsize,labelpad=-12)
plt.ylabel(r'$u_x$',fontsize=fontsize,labelpad=-25)
plt.title('Control',fontsize=fontsize)
font = {'family': 'Times New Roman'
        }
plt.legend(fontsize=ticksize,loc=1,frameon=True,labelcolor='k',handletextpad=0.3,borderpad=0.2,borderaxespad=0.3,handlelength=1.0,prop=font)
plt.text(0,0.6,'(b)',fontsize=fontsize)

ax0 = plt.subplot(131)
plot_grid()
netc = np.concatenate((np.zeros([1]),netc))
d_netc = netc[1:]-netc[:-1]
plt.scatter(netc[1:],d_netc,s=size,marker='o',zorder=2,color=colors[-1],label='NETC: {}'.format(len(d_netc)))
plt.axhline(min(d_netc),color=colors[-1],ls='--',lw=1)
nlc = np.concatenate((np.zeros([1]),nlc))
d_nlc = nlc[1:]-nlc[:-1]
plt.scatter(nlc[1:],d_nlc,s=size,marker='o',zorder=2, c='',edgecolors=colors[-2],label='NLC: {}'.format(len(d_nlc)))
plt.axhline(min(d_nlc),color=colors[-2],ls='--',lw=1)

plt.ylabel('Inter-event time',fontsize=fontsize,labelpad=-18)
plt.xlabel('Time',fontsize=fontsize,labelpad=-12)
plt.xticks([0,nlc[-1]],['0','1'],fontsize=ticksize)
plt.yticks([0,0.4],['0','0.4'],fontsize=ticksize)
plt.ylim(-0.02,0.4)
plt.title('Triggering times',fontsize=fontsize)
plt.legend(fontsize=ticksize,loc=2,frameon=True,labelcolor='k',handletextpad=0.3,borderpad=0.2,borderaxespad=0.3,handlelength=1.0)
plt.text(0.75,0.32,'(a)',fontsize=fontsize)

ax1 = plt.subplot(133)
plot_grid()
nlc_traj = np.load('nlc_traj_2000.npy')[:500]
netc_traj = np.load('netc_traj.npy')[:500]
plt.plot(np.linspace(0,1,len(netc_traj)),netc_traj,c=colors[-1],label='NETC')
plt.plot(np.linspace(0,1,len(nlc_traj)),nlc_traj,c=colors[-2],label='NLC')


# plt.scatter(netc[1:],d_netc,s=size,marker='o',zorder=2, c='',edgecolors=colors[0],label='Triggering times: {}'.format(len(d_netc)))
# plt.scatter(netc[1:],d_netc,s=size,marker='o',zorder=2,color=colors[2],label='Triggering times: {}'.format(len(d_netc)))
# plt.scatter(np.arange(len(d_netc)),d_netc,s=size,marker='o',)
plt.ylabel(r'$x$',fontsize=fontsize,labelpad=-20)
plt.xlabel('Time',fontsize=fontsize,labelpad=-12)
plt.xticks([0,1.0],['0','1'],fontsize=ticksize)
plt.yticks([-3,0,15],['-3','0','15'],fontsize=ticksize)
plt.ylim(-4,16)
plt.title('Trajectories',fontsize=fontsize)
plt.text(0,12.5,'(c)',fontsize=fontsize)
# plt.legend(fontsize=ticksize,loc=1,frameon=True,labelcolor='k',handletextpad=0.4,borderpad=0.2,borderaxespad=0.3,handlelength=1.0)

# axins = inset_axes(ax1, 0.7, 0.7, loc=2, bbox_to_anchor=(0.15, 0.8),
#                    bbox_transform=ax1.figure.transFigure)
axins = ax1.inset_axes(
    [0.48, 3, 0.5, 7], transform=ax1.transData)
axins.plot(np.linspace(0.6,0.8,100), netc_traj[300:400], color=colors[-1])
axins.plot(np.linspace(0.6,0.8,100), nlc_traj[300:400], color=colors[-2])
axins.set_xticks([])
axins.set_yticks([])
# axins.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.)
# axins.minorticks_on()
# axins.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
# axins.plot(np.linspace(0.5,1.0,250), nlc_traj[250:], color='g', label='Truth', ls=(0, (3, 3)), lw=1.0)
# axins.set_xticks([0.6,0.8],['',''])
# axins.set_yticks([-0.4,0.4,['','']])

xyA = (0.47, 3.3)
xyB = (0.62, -0.2)
coordsA = "data"
coordsB = "data"
con1 = ConnectionPatch(xyA, xyB,
                       coordsA, coordsB,
                       arrowstyle="-",
                       shrinkA=5, shrinkB=5,
                       mutation_scale=30,
                       fc="k", color='k')
ax1.add_artist(con1)

xyA = (1.0, 3.2)
xyB = (0.8, -0.5)
coordsA = "data"
coordsB = "data"
con2 = ConnectionPatch(xyA, xyB,
                       coordsA, coordsB,
                       arrowstyle="-",
                       shrinkA=5, shrinkB=5,
                       mutation_scale=30,
                       fc="k", color='k')
ax1.add_artist(con2)

plt.show()