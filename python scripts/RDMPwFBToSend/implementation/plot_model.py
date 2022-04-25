import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import zipfile
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import os
# import model.Extended_RDMP as RDMP

p = .3
q = .4
d = .6
m = 4
n_block = 1000
alpha = .05
matplotlib.rc('xtick', labelsize=35)
matplotlib.rc('ytick', labelsize=35)
# matplotlib.rc('ylabel', labelsize=40)
# matplotlib.rc('ylabel', labelsize=40)
plt.rcParams['xtick.major.pad'] = '30'
plt.rcParams['ytick.major.pad'] = '30'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# y_loc = (0, 1, 2, 3)
y_loc = (0, 2, 4, 6,8)
# y_label = (r'$\bf{30}$', r'$\bf{70}$', r'$\bf{110}$', r'$\bf{150}$')
y_label = (r'$\bf{30}$', r'$\bf{70}$', r'$\bf{110}$', r'$\bf{150}$',r'$\bf{190}$')


def load_object(filename):
    name=filename.split('.')[0]
    name2=filename.split('/')[2]
 # zf = zipfile.ZipFile(name2+'.zip', 'r')
    #zf = zipfile.ZipFile(filename, 'r')
    with open(filename, 'rb') as input:
         obj = pickle.load(input)
    #obj = pickle.loads(zf.open(name2).read())
    #zf.close()


    # with open(filename, 'rb') as input:
    #     obj = pickle.load(input)
    return obj
def load_object_unzip(filename):
    name=filename.split('.')[0]
    name2=filename.split('/')[2]
    zf = zipfile.ZipFile(filename, 'r')
    # with open(filename, 'rb') as input:
    #      obj = pickle.load(input)
    obj = pickle.loads(zf.open(name2).read())
    #zf.close()


    # with open(filename, 'rb') as input:
    #     obj = pickle.load(input)
    return obj

def plot_all(p_range1, p_range2, l_block_range, values, name, vmin, vmax, title, flag_diff):
    plot_heatmap_separated(p_range1, p_range2, l_block_range, values,  name, vmin, vmax, title,
                           flag_diff)
def generateMeanAfterRev(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    i=item_mean[0:int(l_block/5)+1]
    # i=i[i>=0]
    # if (i.shape[0])==0:
    #     print("E")
    #     item_mean_padded=0
    # else:
    item_mean_padded = np.mean(i)
    return item_mean_padded
def generateMeanAfterRev_minusIgnore(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    i=item_mean[0:int(l_block/5)+1]
    i=i[i>=0]
    if (i.shape[0])==0:
        print("E")
        item_mean_padded=0
    else:
        item_mean_padded = np.mean(i)
    return item_mean_padded
def generateMeanBeforeRev(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    item_mean_padded = np.mean(item_mean[l_block-int(l_block/5)+1:l_block])
    return item_mean_padded
def generateMeanBeforeRev2(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    item_mean_padded = np.mean(item_mean[l_block-4*int(l_block/5)+1:l_block])
    return item_mean_padded
def generateMeanStable(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    item_mean_padded = np.mean(item_mean[0:l_block])
    return item_mean_padded
def find_trial(values_full, values_rdmp, l_block):
    #full
    item_temporal_full = np.mean(np.asarray(np.split(values_full, n_block / 2)), 0)
    item_temporal_rdmp = np.mean(np.asarray(np.split(values_rdmp, n_block / 2)), 0)

    indices =np.where((item_temporal_rdmp[:l_block]>=item_temporal_full[:l_block])&(item_temporal_rdmp[:l_block]>0.5))

    if (len(indices[0]) == 0):
        return -50
    # item_mean_padded = np.mean(item_mean[0:l_block])
    else:
        # print(len(indices),indices)
        return indices[0][0]
def generateMeanStable_temporal(values, l_block_max):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    padded_mean = np.pad(item_mean, (0, l_block_max * 2 - item_mean.shape[0]), 'constant',
                         constant_values=(0, 0))
    item_mean_padded = padded_mean
    return item_mean_padded


def plot_heatmap_separated(p1_range, p2_range, l_block_range, values, name, vmin=None, vmax=None, title='',
                           flag_diff=False):

    values1 = np.round(values[:-2, 1:6],2)+0
    values2 = np.round(values[:-2, 6:],2)+0
    # vmin=0
    # vmin=min(np.min(values1),np.min(values2))
    # vmax = max(np.max(values1), np.max(values2))
    fig, ax0 = plt.subplots(1, 2, figsize=(40, 20), sharey=True)
    #cmap = plt.cm.get_cmap("jet")
    cmap = plt.cm.get_cmap("Blues")
    # cmap.set_bad(color='black')
    # cmap.set_under(color='black')
    # cmap = plt.cm.get_cmap("cividis", 256)
    p_range1 = p1_range.tolist()
    p_range2 = p2_range.tolist()
    p_range1 = ['{:.1f}'.format(x) + "/" + '{:.1f}'.format(x) for (x, y) in zip(p_range1, p_range2)]
    l_range = l_block_range.tolist()

    ax0[0].set_yticks(y_loc, minor=False)
    ax0[0].set_yticklabels(y_label, minor=False)

    ax0[0].set_xticks(np.arange(values1.shape[1]), minor=False)
    ax0[0].set_xticklabels((r'$\bf{0.3/0.1}$',
                            r'$\bf{0.4/0.2}$', r'$\bf{0.6/0.2}$',
                            r'$\bf{0.9/0.3}$', r'$\bf{0.9/0.6}$',
                            ), minor=False, rotation=90)
    im = ax0[0].imshow(values1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax0[0].invert_yaxis()
    ax0[1].axes.yaxis.set_visible(False)
    ax0[1].set_xticks((0, 2, 4, 6), minor=False)
    ax0[1].set_xticklabels((r'$\bf{0.6/0.4}$', r'$\bf{0.7/0.3}$',
                            r'$\bf{0.8/0.2}$', r'$\bf{0.9/0.1}$'), minor=False, rotation=90)
    im = ax0[1].imshow(values2, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax0[1].invert_yaxis()
    plt.xlabel(r'$\bf{uncertain  \longleftarrow p_{R}(B)/p_{R}(w)  \longrightarrow  certain}$', color='k', size=50,
               labelpad=40)

    # fig.subplots_adjust(wspace=0.1, left=0.02, bottom=0.2, top=0.9)
    fig.subplots_adjust( bottom=0.2, top=0.9)
    ax0[0].tick_params(which='both', width=5)
    ax0[0].tick_params(which='major', length=14)
    ax0[1].tick_params(which='both', width=5)
    ax0[1].tick_params(which='major', length=14)

    for i in range(values1.shape[0]):
        for j in range(values1.shape[1]):
            text = ax0[0].text(j, i,   '%.2f' % values1[i, j],
                           ha="center", va="center", color="w",size=35)


    for i in range(values2.shape[0]):
        for j in range(values2.shape[1]):
            text = ax0[1].text(j, i,  '%.2f' % values2[i, j],
                           ha="center", va="center", color="w",size=35)



    cs = plt.cm.ScalarMappable(cmap=cmap)
    cs.set_array(values)
    cs.set_clim(vmin, vmax)

    if (not flag_diff):
        cbar = fig.colorbar(cs, ax=ax0, ticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        cbar.ax.set_yticklabels(
            [r'$\bf{0.5}$', r'$\bf{0.6}$', r'$\bf{0.7}$', r'$\bf{0.8}$', r'$\bf{0.9}$'], size=35)
        cbar.ax.tick_params(which='both', width=5)
        cbar.ax.tick_params(which='major', length=14)


    else:
        cbar = fig.colorbar(cs, ax=ax0)
        cbar.ax.tick_params(which='both', width=5)
        cbar.ax.tick_params(which='major', length=14)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")

    fig.supylabel(r'$\bf{Volatie  \longleftarrow L  \longrightarrow  Stable}$', color='k', size=50, x=0.02)
    fig.suptitle(title, fontsize=40, fontweight='bold', x=0.4)
    # plt.colorbar(cs)
    plt.savefig(folder + name + '.png')
    # plt.savefig("../results/" + name + '.pdf')
    plt.close(fig)
def plot_heatmap_separated_two(p1_range, p2_range, l_block_range, orig_values1,orig_values2, name, vmin=None, vmax=None, title='',
                           flag_diff=False):

    values1_1 = np.round(orig_values1[:-2, 1:6],2)+0
    values1_2 = np.round(orig_values1[:-2, 6:],2)+0

    values2_1 = np.round(orig_values2[:-2, 1:6], 2) + 0
    values2_2 = np.round(orig_values2[:-2, 6:], 2) + 0
    # vmin=0
    # vmin=min(np.min(values1),np.min(values2))
    # vmax = max(np.max(values1), np.max(values2))
    fig, ax0 = plt.subplots(1, 2, figsize=(40, 20), sharey=True)
    cmap = plt.cm.get_cmap("Blues")
    # cmap = plt.cm.get_cmap("cividis", 256)
    p_range1 = p1_range.tolist()
    p_range2 = p2_range.tolist()
    p_range1 = ['{:.1f}'.format(x) + "/" + '{:.1f}'.format(x) for (x, y) in zip(p_range1, p_range2)]
    l_range = l_block_range.tolist()

    ax0[0].set_yticks(y_loc, minor=False)
    ax0[0].set_yticklabels(y_label, minor=False)

    ax0[0].set_xticks(np.arange(values1_1.shape[1]), minor=False)
    ax0[0].set_xticklabels((r'$\bf{0.3/0.1}$',
                            r'$\bf{0.4/0.2}$', r'$\bf{0.6/0.2}$',
                            r'$\bf{0.9/0.3}$', r'$\bf{0.9/0.6}$',
                            ), minor=False, rotation=90)
    im = ax0[0].imshow(values1_1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax0[0].invert_yaxis()
    ax0[1].axes.yaxis.set_visible(False)
    ax0[1].set_xticks((0, 2, 4, 6), minor=False)
    ax0[1].set_xticklabels((r'$\bf{0.6/0.4}$', r'$\bf{0.7/0.3}$',
                            r'$\bf{0.8/0.2}$', r'$\bf{0.9/0.1}$'), minor=False, rotation=90)
    im = ax0[1].imshow(values1_2, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax0[1].invert_yaxis()
    plt.xlabel(r'$\bf{uncertain  \longleftarrow p_{R}(B)/p_{R}(w)  \longrightarrow  certain}$', color='k', size=50,
               labelpad=40)

    # fig.subplots_adjust(wspace=0.1, left=0.02, bottom=0.2, top=0.9)
    fig.subplots_adjust( bottom=0.2, top=0.9)
    ax0[0].tick_params(which='both', width=5)
    ax0[0].tick_params(which='major', length=14)
    ax0[1].tick_params(which='both', width=5)
    ax0[1].tick_params(which='major', length=14)

    # for i in range(values1.shape[0]):
    #     for j in range(values1.shape[1]):
    #         text = ax0[0].text(j, i,   '%.2f' % values1[i, j],
    #                        ha="center", va="center", color="w",size=35)
    #
    #
    # for i in range(values2.shape[0]):
    #     for j in range(values2.shape[1]):
    #         text = ax0[1].text(j, i,  '%.2f' % values2[i, j],
    #                        ha="center", va="center", color="w",size=35)



    cs = plt.cm.ScalarMappable(cmap=cmap)
    cs.set_array(values)
    cs.set_clim(vmin, vmax)

    if (not flag_diff):
        cbar = fig.colorbar(cs, ax=ax0, ticks=[0.5, 0.6, 0.7, 0.8, 0.9])
        cbar.ax.set_yticklabels(
            [r'$\bf{0.5}$', r'$\bf{0.6}$', r'$\bf{0.7}$', r'$\bf{0.8}$', r'$\bf{0.9}$'], size=35)
        cbar.ax.tick_params(which='both', width=5)
        cbar.ax.tick_params(which='major', length=14)


    else:
        cbar = fig.colorbar(cs, ax=ax0)
        cbar.ax.tick_params(which='both', width=5)
        cbar.ax.tick_params(which='major', length=14)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")

    fig.supylabel(r'$\bf{Volatie  \longleftarrow L  \longrightarrow  Stable}$', color='k', size=50, x=0.02)
    fig.suptitle(title, fontsize=40, fontweight='bold', x=0.4)
    # plt.colorbar(cs)
    plt.savefig(folder + name + '.png')
    # plt.savefig("../results/" + name + '.pdf')
    plt.close(fig)


def plot_temporal(k,rpe,delta_f,diff,l_block, p_range1, p_range2, title, flag):
    fig1, ax = plt.subplots(p_range1.shape[0]//2+1, 2, figsize=(30, 50))
    fig1.tight_layout()  #
    plt.subplots_adjust(hspace=0.6)
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(bottom=0.07)
    # k=
    for i, prob in enumerate(p_range1):
        l_block = int(l_block)
        if i<=p_range1.shape[0]//2:
            j=0
            l=i
        else:
            j=1
            l=i%(p_range1.shape[0]//2)
        ax[l,j].plot(np.arange(0, 2 * l_block), rpe[i, :2 * l_block], linewidth=3,
                      color="red",
                      # linestyle='-', label=r'$\Delta_F$')
                      linestyle='-', label='rpe')

        ax[l,j].plot(np.arange(0, 2 * l_block), delta_f[i, :2 * l_block], linewidth=3,
                      color="blue",
                      # linestyle='-', label=r'$\Delta_F$')
                      linestyle='-', label='delta_f')

        ax[l,j].plot(np.arange(0, 2 * l_block), diff[i, :2 * l_block], linewidth=3,
                      color="black",
                      # linestyle='-', label=r'$\Delta_F$')
                      linestyle='-', label='diff')
        ax[l,j].set_title(
            str(l_block) + " |" + "{:.2f}".format(prob) + "/" + str(p_range2[i]) + "\n",
            fontsize='30', fontweight='bold')
        ax[l,j].set_xlabel("trial", fontsize='20', fontweight='bold')
        # if flag:
        #     # m1=np.min(diff[i, j])
        #     # m2=np.min(exp_p[i, j])
        #     # m3=np.min(diff[i, j]*exp_p[i, j])
        #     # m4=np.min(sum[i,j]*exp_p[i, j])
        #     m5=np.min(x[i, j])
        #     min_y= min(m5,0)
        #
        #     # m1 = np.max(diff[i, j])
        #     # m2 = np.max(exp_p[i, j])
        #     # m5 = np.max(exp_p[i, j])
        #     # m3=np.max(diff[i, j]*exp_p[i, j])
        #     m5=np.max(x[i, j])
        #
        #     max_y= m5
        #     decimal.getcontext().rounding = decimal.ROUND_CEILING
        #     max_y= float(round(decimal.Decimal(str(max_y)), ndigits=1))
        #     # print(max_y)
        #     ax[i, j].set_ylim(min_y,max_y)
        #     # print(int(max_y / 0.1))
        #
        #     # here's the one-liner
        #
        #
        #     y_labels = np.linspace(min_y, max_y, int((float(round(decimal.Decimal(str((max_y-min_y) / 0.1)), ndigits=1))+1)))
        #     print(y_labels)
        #     ax[i, j].set_yticklabels(np.around(y_labels,1), fontsize='24', fontweight='bold')
        ax[l,j].set_xlim(l_block, l_block + 5)
        ax[l,j].set_xticks(np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int))
        x_labels = np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int)
        ax[l,j].set_xticklabels(x_labels, fontsize='24', fontweight='bold')
        # ax[i,(i)%2].vlines(l_block, linestyles="dashed", colors="red", ymin=0,ymax=2)
        ax[l,j].grid(True)

    legend_properties = {'size': '30'}
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', prop=legend_properties,ncol=3)
    # txt = "Decision Making Based on The Estimation of Choice Probability from Plastic Synapses, p="+str(p)+"q="+str(q)
    # plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment='center', fontsize=30,bbox=dict(boxstyle='round',ec="k",color='#f5e9f2'))

    fig1.savefig(
        folder + title +"rpe_temporal.png")
    plt.ylabel("rpe", fontsize='20', fontweight='bold')
def plot_temporal_two(k, full,rdmp, l_block, p_range1, p_range2, title, flag):
    fig1, ax = plt.subplots(p_range1.shape[0]//2+1, 2, figsize=(30, 50))
    fig1.tight_layout()  #
    plt.subplots_adjust(hspace=0.6)
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(bottom=0.07)
    # k=
    for i, prob in enumerate(p_range1):
        l_block = int(l_block)
        if i<=p_range1.shape[0]//2:
            j=0
            l=i
        else:
            j=1
            l=i%(p_range1.shape[0]//2)
        ax[l,j].plot(np.arange(0, 2 * l_block), full[i, :2 * l_block], linewidth=3,
                     color="red",
                     # linestyle='-', label=r'$\Delta_F$')
                     linestyle='-', label=title)

        ax[l,j].plot(np.arange(0, 2 * l_block), rdmp[i, :2 * l_block], linewidth=3,
                     color="blue",
                     # linestyle='-', label=r'$\Delta_F$')
                     linestyle='-', label=title)


        ax[l,j].set_title(
            str(l_block) + " |" + "{:.2f}".format(prob) + "/" + str(p_range2[i]) + "\n",
            fontsize='30', fontweight='bold')
        ax[l,j].set_xlabel("trial", fontsize='20', fontweight='bold')
        # if flag:
        #     # m1=np.min(diff[i, j])
        #     # m2=np.min(exp_p[i, j])
        #     # m3=np.min(diff[i, j]*exp_p[i, j])
        #     # m4=np.min(sum[i,j]*exp_p[i, j])
        #     m5=np.min(x[i, j])
        #     min_y= min(m5,0)
        #
        #     # m1 = np.max(diff[i, j])
        #     # m2 = np.max(exp_p[i, j])
        #     # m5 = np.max(exp_p[i, j])
        #     # m3=np.max(diff[i, j]*exp_p[i, j])
        #     m5=np.max(x[i, j])
        #
        #     max_y= m5
        #     decimal.getcontext().rounding = decimal.ROUND_CEILING
        #     max_y= float(round(decimal.Decimal(str(max_y)), ndigits=1))
        #     # print(max_y)
        #     ax[i, j].set_ylim(min_y,max_y)
        #     # print(int(max_y / 0.1))
        #
        #     # here's the one-liner
        #
        #
        #     y_labels = np.linspace(min_y, max_y, int((float(round(decimal.Decimal(str((max_y-min_y) / 0.1)), ndigits=1))+1)))
        #     print(y_labels)
        #     ax[i, j].set_yticklabels(np.around(y_labels,1), fontsize='24', fontweight='bold')
        ax[l,j].set_xlim(l_block, l_block + 5)
        ax[l,j].set_xticks(np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int))
        x_labels = np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int)
        ax[l,j].set_xticklabels(x_labels, fontsize='24', fontweight='bold')
        # ax[i,(i)%2].vlines(l_block, linestyles="dashed", colors="red", ymin=0,ymax=2)
        ax[l,j].grid(True)

    legend_properties = {'size': '30'}
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', prop=legend_properties,ncol=3)
    # txt = "Decision Making Based on The Estimation of Choice Probability from Plastic Synapses, p="+str(p)+"q="+str(q)
    # plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment='center', fontsize=30,bbox=dict(boxstyle='round',ec="k",color='#f5e9f2'))

    fig1.savefig(
        folder + title +"rpe_temporal.png")
    plt.ylabel("rpe", fontsize='20', fontweight='bold')
def plot_temporal_one(k, full, l_block, p_range1, p_range2, title, flag):
    fig1, ax = plt.subplots(p_range1.shape[0]//2+1, 2, figsize=(30, 50))
    fig1.tight_layout()  #
    plt.subplots_adjust(hspace=0.6)
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(bottom=0.07)
    # k=
    for i, prob in enumerate(p_range1):
        l_block = int(l_block)
        if i<=p_range1.shape[0]//2:
            j=0
            l=i
        else:
            j=1
            l=i%(p_range1.shape[0]//2)
        ax[l,j].plot(np.arange(0, 2 * l_block), full[i, :2 * l_block], linewidth=3,
                     color="red",
                     # linestyle='-', label=r'$\Delta_F$')
                     linestyle='-', label=title)


        ax[l,j].set_title(
            str(l_block) + " |" + "{:.2f}".format(prob) + "/" + str(p_range2[i]) + "\n",
            fontsize='30', fontweight='bold')
        ax[l,j].set_xlabel("trial", fontsize='20', fontweight='bold')
        # if flag:
        #     # m1=np.min(diff[i, j])
        #     # m2=np.min(exp_p[i, j])
        #     # m3=np.min(diff[i, j]*exp_p[i, j])
        #     # m4=np.min(sum[i,j]*exp_p[i, j])
        #     m5=np.min(x[i, j])
        #     min_y= min(m5,0)
        #
        #     # m1 = np.max(diff[i, j])
        #     # m2 = np.max(exp_p[i, j])
        #     # m5 = np.max(exp_p[i, j])
        #     # m3=np.max(diff[i, j]*exp_p[i, j])
        #     m5=np.max(x[i, j])
        #
        #     max_y= m5
        #     decimal.getcontext().rounding = decimal.ROUND_CEILING
        #     max_y= float(round(decimal.Decimal(str(max_y)), ndigits=1))
        #     # print(max_y)
        #     ax[i, j].set_ylim(min_y,max_y)
        #     # print(int(max_y / 0.1))
        #
        #     # here's the one-liner
        #
        #
        #     y_labels = np.linspace(min_y, max_y, int((float(round(decimal.Decimal(str((max_y-min_y) / 0.1)), ndigits=1))+1)))
        #     print(y_labels)
        #     ax[i, j].set_yticklabels(np.around(y_labels,1), fontsize='24', fontweight='bold')
        ax[l,j].set_xlim(l_block, l_block + 5)
        ax[l,j].set_xticks(np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int))
        x_labels = np.arange(0, int(2 * l_block + 1), step=l_block / 6, dtype=int)
        ax[l,j].set_xticklabels(x_labels, fontsize='24', fontweight='bold')
        # ax[i,(i)%2].vlines(l_block, linestyles="dashed", colors="red", ymin=0,ymax=2)
        ax[l,j].grid(True)

    legend_properties = {'size': '30'}
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', prop=legend_properties,ncol=3)
    # txt = "Decision Making Based on The Estimation of Choice Probability from Plastic Synapses, p="+str(p)+"q="+str(q)
    # plt.figtext(0.5, 0.02, txt, wrap=True, horizontalalignment='center', fontsize=30,bbox=dict(boxstyle='round',ec="k",color='#f5e9f2'))

    fig1.savefig(
        folder + title +"rpe_temporal.png")
    plt.ylabel("rpe", fontsize='20', fontweight='bold')


def plot_eu():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    # l_block_range = np.arange(30, 180, 40)

    eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])


    eu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])


    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    #model = load_object(folder+"modelll_rdmp2_1000.pkl")
    #model = (folder+"modelll_rdmp2_1000.zip")
    model = load_object(folder+"modelll_rdmp2_1000.pkl")




    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            eu[i, j] = generateMeanStable(model_rdmp.rpe, l_block)
            eu_meta[i, j] = generateMeanStable(model_rdmp.rpe_meta, l_block)

            eu_before[i, j] = generateMeanBeforeRev(model_rdmp.rpe, l_block)
            eu_meta_before[i, j] = generateMeanBeforeRev(model_rdmp.rpe_meta, l_block)

            eu_before2[i, j] = generateMeanBeforeRev2(model_rdmp.rpe, l_block)
            eu_meta_before2[i, j] = generateMeanBeforeRev2(model_rdmp.rpe_meta, l_block)

            eu_after[i, j] = generateMeanAfterRev(model_rdmp.rpe, l_block)
            eu_meta_after[i, j] = generateMeanAfterRev(model_rdmp.rpe_meta, l_block)

    #Figure 5
    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_before), "eu_all_before_",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Plastic Synapses before Reversal ', flag_diff=True)

    #Figure 6
    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_meta_before2), "eu_meta_all_before",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Meta-plastic Synapses before Reversal', flag_diff=True)



def plot_eu_full():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    # l_block_range = np.arange(30, 180, 40)

    eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])


    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_full2_1000.pkl")



    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            eu[i, j] = generateMeanStable(model_rdmp.rpe, l_block)
            eu_meta[i, j] = generateMeanStable(model_rdmp.rpe_meta, l_block)

            eu_before[i, j] = generateMeanBeforeRev(model_rdmp.rpe, l_block)
            eu_meta_before[i, j] = generateMeanBeforeRev(model_rdmp.rpe_meta, l_block)

            eu_after[i, j] = generateMeanAfterRev(model_rdmp.rpe, l_block)
            eu_meta_after[i, j] = generateMeanAfterRev(model_rdmp.rpe_meta, l_block)


    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu), "eu_all_full",
             # vmin=min(np.min(eu), np.min(eu_meta)),
             # vmax=max(np.max(eu), np.max(eu_meta)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Plastic Synapses', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_meta), "eu_meta_all_full",
             # vmin=min(np.min(eu), np.min(eu_meta)), vmax=max(np.max(eu), np.max(eu_meta)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Meta-plastic Synapses', flag_diff=True)


    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_before), "eu_all_before_full",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Plastic Synapses before Reversal ', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_meta_before), "eu_meta_all_before_full",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Meta-plastic Synapses before Reversal', flag_diff=True)



    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_after), "eu_all_after_full",
             vmin=.1,
             vmax=.6,
             title='Expected Uncertainty from Plastic Synapses after Reversal', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_meta_after), "eu_meta_all_after_full",
             # vmin=min(np.min(eu_after), np.min(eu_meta_after)),
             # vmax=max(np.max(eu_after), np.max(eu_meta_after)),
             vmin=.1,
             vmax=.6,
             title='Expected Uncertainty from Meta-plastic Synapses after Reversal', flag_diff=True)
def plot_rpe(k):

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])


    l_block_range = np.arange(30,200,20)
    rpe = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    delta_f = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    diff = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])

   # [30,50,70,90,110,130,150,170,190]

    # eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # eu_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    #
    # eu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # eu_meta_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    #
    # eu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # eu_meta_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])


    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_rdmp2_1000.pkl")



    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            l_block = int(l_block)
            print(i, j)
            model_rdmp = model[i][j]
            rpe[i,j]= generateMeanStable_temporal(model_rdmp.rpe_avg_meanRmv, np.max(l_block_range))
            delta_f[i,j]= generateMeanStable_temporal(model_rdmp.deltaF, np.max(l_block_range))
            diff[i,j]= generateMeanStable_temporal(model_rdmp.dif_meanRemov, np.max(l_block_range))

    # plot_temporal(j, rpe, p_range1, p_range2, l_block_range, title, flag):
    plot_temporal(k,rpe[:,k,:],delta_f[:,k,:],diff[:,k,:],l_block_range[k], p_range1, p_range2, title='RPE_'+str(l_block_range[k]), flag=True)
    # plot_temporal(k,delta_f[:,k,:],l_block_range[k], p_range1, p_range2, title='deltaf_'+str(l_block_range[k]), flag=True)


def plot_uu():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)

    uu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_rdmp2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            uu[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            uu_before[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            uu_before2[i, j] = generateMeanBeforeRev2(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            uu_after[i, j] = generateMeanAfterRev(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)

    #Figure 7
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_before2), "uu_before2",
             # vmin=np.min(uu_after), vmax=np.max(uu_after),
             vmin=.1, vmax=.24,
             title='Unexpected Uncertainty from Meta-plastic Synapses brefore Reversal' , flag_diff=True)
def plot_uu_full():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)

    uu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    #model = load_object(folder+"modelll_full2_1000.zip")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            uu[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            uu_before[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            uu_after[i, j] = generateMeanAfterRev(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu), "uu_full",
             vmin=np.min(uu), vmax=np.max(uu),
             # vmin=.1, vmax=.24,
             title='Unexpected Uncertainty from Meta-plastic Synapses', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_before), "uu_before_full",
             vmin=np.min(uu_before), vmax=np.max(uu_before),
             # vmin=.1, vmax=.24,
             title='Unexpected Uncertainty from Meta-plastic Synapses', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_after), "uu_after_full",
             vmin=np.min(uu_after), vmax=np.max(uu_after),
             # vmin=.1, vmax=.24,
             title='Unexpected Uncertainty from Meta-plastic Synapses', flag_diff=True)
def plot_uu_minus_eu_rdmp():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)
    # l_block_range = np.arange(30, 180, 40)

    uu_minus_eu_meanRemoved = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_minus_eu_meanRemoved2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    # uu_minus_eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_rdmp2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            #uu_minus_eu_meanRemoved[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)

            uu_minus_eu_meanRemoved2[i, j] = generateMeanBeforeRev2(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    # plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_meanRemoved_rdmp_beforeRev",
    #          # vmin=np.min(uu_minus_eu_meanRemoved), vmax=np.max(uu_minus_eu_meanRemoved),
    #          vmin=0, vmax=.1,
    #          title='Difference between Unexpected Uncertainty and Expected Uncertainty before Reversal', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved2), "uu_minus_eu_meanRemoved_rdmp_beforeRev2",
             # vmin=np.min(uu_minus_eu_meanRemoved), vmax=np.max(uu_minus_eu_meanRemoved),
             vmin=0, vmax=.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty before Reversal', flag_diff=True)



    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            # uu_minus_eu[i, j] = generateMeanAfterRev(np.subtract(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w) , model_rdmp.rpe_avg), l_block)
            uu_minus_eu_meanRemoved[i, j] = generateMeanAfterRev_minusIgnore(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    # plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu), "uu_minus_eu_afterRev", vmin=np.min(uu_minus_eu), vmax=np.max(uu_minus_eu),
    #          title='Difference between Expected Uncertainty and Unexpected Uncertainty', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_meanRemoved_rdmp_afterRev",
             # vmin=np.min(uu_minus_eu_meanRemoved), vmax=0.1,
             vmin=0, vmax=0.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty after Reversal', flag_diff=True)


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            # uu_minus_eu[i, j] = generateMeanStable(np.subtract(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w) , model_rdmp.rpe_avg), l_block)
            uu_minus_eu_meanRemoved[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    # plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu), "uu_minus_eu_all", vmin=np.min(uu_minus_eu), vmax=np.max(uu_minus_eu),
    #          title='Difference between Expected Uncertainty and Unexpected Uncertainty', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_rdmp_meanRemoved_all",
             # vmin=np.min(uu_minus_eu_meanRemoved), vmax=np.max(uu_minus_eu_meanRemoved),
             vmin=0, vmax=.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)
def plot_uu_minus_eu_rdmp_NotMeanRemoveFromEU():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)
    # l_block_range = np.arange(30, 180, 40)

    uu_minus_eu_meanRemoved_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_minus_eu_meanRemoved_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_minus_eu_meanRemoved_stable = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    uu_minus_eu_meta_meanRemoved_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_minus_eu_meta_meanRemoved_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    uu_minus_eu_meta_meanRemoved_stable = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    # uu_minus_eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_rdmp2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            uu_minus_eu_meanRemoved_before[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF , model_rdmp.rpe), l_block)
            uu_minus_eu_meanRemoved_after[i, j] = generateMeanAfterRev(np.subtract(model_rdmp.deltaF , model_rdmp.rpe), l_block)
            uu_minus_eu_meanRemoved_stable[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF , model_rdmp.rpe), l_block)

            uu_minus_eu_meta_meanRemoved_before[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF, model_rdmp.rpe_meta), l_block)
            uu_minus_eu_meta_meanRemoved_after[i, j] = generateMeanAfterRev(np.subtract(model_rdmp.deltaF, model_rdmp.rpe_meta),l_block)
            uu_minus_eu_meta_meanRemoved_stable[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF, model_rdmp.rpe_meta), l_block)


            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved_before), "uu_minus_eu_rdmp_beforeRev",
             vmin=np.min(uu_minus_eu_meanRemoved_before), vmax=np.max(uu_minus_eu_meanRemoved_before),
             title='Difference between Expected Uncertainty and Unexpected Uncertainty before Reversal', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved_after), "uu_minus_eu_rdmp_afterRev",
             vmin=np.min(uu_minus_eu_meanRemoved_after), vmax=np.max(uu_minus_eu_meanRemoved_after),
             title='Difference between Unexpected Uncertainty and Expected Uncertainty after Reversal', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved_stable),
             "uu_minus_eu_rdmp_all",
             vmin=np.min(uu_minus_eu_meanRemoved_stable), vmax=np.max(uu_minus_eu_meanRemoved_stable),
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)


    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meta_meanRemoved_before), "uu_minus_eu_meta_rdmp_beforeRev",
             vmin=np.min(uu_minus_eu_meta_meanRemoved_before), vmax=np.max(uu_minus_eu_meta_meanRemoved_before),
             title='Difference between Unexpected Uncertainty and Expected Uncertainty before Reversal', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meta_meanRemoved_after), "uu_minus_eu_meta_rdmp_afterRev",
             vmin=np.min(uu_minus_eu_meta_meanRemoved_after), vmax=np.max(uu_minus_eu_meta_meanRemoved_after),
             title='Difference between Unexpected Uncertainty and Expected Uncertainty after Reversal', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meta_meanRemoved_stable),
             "uu_minus_eu_meta_rdmp_all",
             vmin=np.min(uu_minus_eu_meta_meanRemoved_stable), vmax=np.max(uu_minus_eu_meta_meanRemoved_stable),
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)



def plot_uu_minus_eu_full():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)
    # l_block_range = np.arange(30, 180, 40)

    uu_minus_eu_meanRemoved = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    # uu_minus_eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model = load_object(folder+"modelll_full2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            uu_minus_eu_meanRemoved[i, j] = generateMeanBeforeRev(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_meanRemoved_full_beforeRev", vmin=-0.1, vmax=0.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)



    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            # uu_minus_eu[i, j] = generateMeanAfterRev(np.subtract(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w) , model_rdmp.rpe_avg), l_block)
            uu_minus_eu_meanRemoved[i, j] = generateMeanAfterRev(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    # plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu), "uu_minus_eu_afterRev", vmin=np.min(uu_minus_eu), vmax=np.max(uu_minus_eu),
    #          title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_meanRemoved_full_afterRev", vmin=-0.1, vmax=0.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_rdmp = model[i][j]
            # uu_minus_eu[i, j] = generateMeanStable(np.subtract(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w) , model_rdmp.rpe_avg), l_block)
            uu_minus_eu_meanRemoved[i, j] = generateMeanStable(np.subtract(model_rdmp.deltaF , model_rdmp.rpe_avg_meanRmv), l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    # plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu), "uu_minus_eu_all", vmin=np.min(uu_minus_eu), vmax=np.max(uu_minus_eu),
    #          title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_minus_eu_meanRemoved), "uu_minus_eu_full_meanRemoved_all", vmin=-0.1, vmax=0.1,
             title='Difference between Unexpected Uncertainty and Expected Uncertainty', flag_diff=True)
def plot_performance():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([ .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([ .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    # l_block_range = np.arange(30,200,20)
    # l_block_range = np.arange(30,120,10)
    l_block_range = np.arange(30, 200, 20)
    # l_block_range = np.arange(30, 180, 40)


    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # p1p_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_rdmp2_1000.pkl")


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_rdmp = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_rdmp[i, j] = generateMeanStable(model_rdmp.p_acc[1] , l_block)



    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full-performance_rdmp), "rdmpWf_rdmp_performance",
             vmin=np.min(performance_full[1:,:-2] -performance_rdmp[1:,:-2] ), vmax=np.max(performance_full[1:,:-2] -performance_rdmp[1:,:-2] ),
             title='Difference between Performance of the RDMPwFB and RDMP Model', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range,
             np.transpose(performance_rdmp-performance_full), "rdmp_rdmpWf_performance",
             vmin=np.min(performance_rdmp[1:,:-2]-performance_full[1:,:-2]), vmax=np.max(performance_rdmp[1:,:-2]-performance_full[1:,:-2]),
             title='Differnce between Performance of the RDMP and RDMPwFB Model', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose((performance_full - performance_rdmp) / performance_rdmp),
             "rdmpWf_rdmp_normalized_performance",
             vmin=np.min((performance_full[1:,:-2] - performance_rdmp[1:,:-2] ) / performance_rdmp[1:,:-2] ), vmax=np.max((performance_full[1:,:-2]   - performance_rdmp[1:,:-2] ) / performance_rdmp[1:,:-2]  ),
             title='Normalized Difference between Performance of the RDMPwFB and RDMP Model', flag_diff=True)


def plot_performance_normalized():
    p_range1 = np.asarray([ .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([ .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)


    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_rdmp2_1000.pkl")


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_rdmp = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_rdmp[i, j] = generateMeanStable(model_rdmp.p_acc[1] , l_block)


    plot_all(p_range1, p_range2, l_block_range, np.transpose((performance_full-performance_rdmp)/performance_rdmp), "rdmpWf_rdmp_normalized_performance",
             vmin=None, vmax=None,
             title='Normalized Difference between Performance of the RDMPwFB and RDMP Model', flag_diff=True)

def plot_performance_full_random():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)
    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_random2_1000.pkl")


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "rdmpWrf_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the RDMP Model with Random Feedback', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full-performance_random), "rdmpWf_rdmpWranf_performance", vmin=np.min(performance_full-performance_random), vmax=np.max(performance_full-performance_random),
             title='Difference between Performance of the RDMPwFB and \n Model with Random Feedback', flag_diff=True)
def plot_performance_full_plastic():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)


    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_plastic_1000.pkl")


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "plastic_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the Plastic Model', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random-performance_full), "rdmpWf_plastic_performance",
             vmin=-.12, vmax=.0,
             title='Difference between Performance of the RDMPwFB and \n a Plastic Model', flag_diff=True)
def plot_performance_full_static():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)


    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")

    th=0
    model_r = load_object(folder+"modelll_staticth"+str(th)+"_1000.pkl")
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "rdmpWst"+str(th)+"_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the RDMP Model with Static threshold', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random-performance_full), "rdmpWf_rdmpWst"+str(th)+"_performance",
             vmin=-0.04, vmax=0.04,
             title='Difference between Performance RDMPwFB with Lesion in Striatum and static threshold=' + str(th)
                   + '\n and Performance of the RDMPwFB with Dynamic threshold', flag_diff=True)


    th='001'
    th1=0.01

    model_r = load_object(folder+"modelll_staticth"+str(th)+"_1000.pkl")
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "rdmpWst"+str(th)+"_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the RDMP Model with Static threshold', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random-performance_full), "rdmpWf_rdmpWst"+str(th)+"_performance",
             # vmin=np.min(performance_full-performance_random), vmax=np.max(performance_full-performance_random),
             vmin=-0.04, vmax=0.04,
             title='Difference between Performance RDMPwFB with Lesion in Striatum and static threshold=' + str(th1)
                   + '\n and Performance of the RDMPwFB with Dynamic threshold', flag_diff=True)

    th='002'
    th1=0.02
    model_r = load_object(folder+"modelll_staticth"+str(th)+"_1000.pkl")
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "rdmpWst"+str(th)+"_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the RDMP Model with Static threshold', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random-performance_full), "rdmpWf_rdmpWst"+str(th)+"_performance",
             # vmin=np.min(performance_full-performance_random), vmax=np.max(performance_full-performance_random),
             vmin=-0.04, vmax=0.04,
             title='Difference between Performance RDMPwFB with Lesion in Striatum and static threshold=' + str(th1)
                   + '\n and Performance of the RDMPwFB with Dynamic threshold', flag_diff=True)


def plot_performance_full_euOnly():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30, 200, 20)

    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")

    th='04'
    th1=0.4

    model_r = load_object(folder+"modelll_euOnlyth"+str(th)+"_1000.pkl")
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_random = model_r[i][j]
            performance_full[i, j] = generateMeanStable(model_full.p_acc[1] , l_block)
            performance_random[i, j] = generateMeanStable(model_random.p_acc[1] , l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose([performance_random]), "rdmpWEu"+str(th)+"_performance", vmin=np.min(performance_random), vmax=np.max(performance_random),
             title='Performance of the RDMP Model with Static threshold', flag_diff=True)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random-performance_full), "rdmpWf_rdmpWEu"+str(th)+"_performance",
             # vmin=np.min(performance_random-performance_full), vmax=np.max(performance_random-performance_full),
             vmin=-0.06, vmax=0.04,
             title='Difference between Performance RDMPwFB with Lesion in ACC to BLA and static threshold=' + str(th1)
                   +'\n and Performance of the RDMPwFB with Dynamic threshold' , flag_diff=True)

def plot_prob_dist():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30, 200, 20)

    performance_full_temporal = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    performance_full_all = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_full_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_full_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            performance_full_all[i, j] = generateMeanStable(model_full.destablization_count , l_block)
            performance_full_before[i, j] = generateMeanBeforeRev2(model_full.destablization_count , l_block)
            performance_full_after[i, j] = generateMeanAfterRev(model_full.destablization_count , l_block)
            performance_full_temporal[i, j] =  generateMeanStable_temporal(model_full.destablization_count, np.max(l_block_range))

    # plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_all), "rdmpWf_dest_all",
    #          # vmin=0, vmax=.8,
    #          vmin=0, vmax=.8,
    #          title='Probability of Destabilization', flag_diff=True)

    #Figure 8
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_before), "rdmpWf_dest_before",
             # vmin=0, vmax=.8,
             vmin=0, vmax=.1,
             title='Probability of Destabilization before the reversal', flag_diff=True)
    #Figure 9
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_after), "rdmpWf_dest_after",
             # vmin=0, vmax=.8,
             vmin=0, vmax=.1,
             title='Probability of Destabilization after the reversal', flag_diff=True)

    #plot_temporal_one(5,performance_full_temporal[:,5,:],l_block_range[5], p_range1, p_range2, title='dest_'+str(l_block_range[5]), flag=True)
def plot_temporal():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30, 200, 20)

    performance_full_temporal = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    performance_rdmp_temporal = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])

    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_rdmp2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            #model_rdmp = model_f[i][j]

            performance_full_temporal[i, j] =  generateMeanStable_temporal(model_full.deltaF, np.max(l_block_range))
            performance_rdmp_temporal[i, j] =  generateMeanStable_temporal(model_full.rpe_avg_meanRmv, np.max(l_block_range))

    # plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_all), "rdmpWf_dest_all",
    #          # vmin=0, vmax=.8,
    #          vmin=0, vmax=.8,
    #          title='Probability of Destabilization', flag_diff=True)

    #plot_temporal_one(5,performance_full_temporal[:,5,:],l_block_range[5], p_range1, p_range2, title='dest_'+str(l_block_range[5]), flag=True)
            plot_temporal_two(5,performance_full_temporal[:,5:],performance_rdmp_temporal[:,5,:],l_block_range[5], p_range1, p_range2, title='dest2_'+str(l_block_range[-1]), flag=True)



def plot_trial_converge():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)


    performance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],2*np.max(l_block_range)])
    trials = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    model_f = load_object(folder+"modelll_full2_1000.pkl")
    model_r = load_object(folder+"modelll_rdmp2_1000.pkl")

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            print(i, j)

            model_full = model_f[i][j]
            model_rdmp = model_r[i][j]
            performance_full[i, j] =  generateMeanStable_temporal(model_rdmp.p_acc[1], np.max(l_block_range))
            performance_rdmp[i, j] =  generateMeanStable_temporal(model_full.p_acc[1], np.max(l_block_range))
            trials[i, j] =  find_trial(model_full.p_acc[1], model_rdmp.p_acc[1],l_block)
            # p1p_meta[i, j] = generateMean(model_rdmp.p1p_meta, l_block)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(trials), "trials", vmin=0, vmax=100,
             title='', flag_diff=True)

    plot_temporal_two(5,performance_full[:,5,:],performance_rdmp[:,5,:],l_block_range[5], p_range1, p_range2, title='trials2_'+str(l_block_range[5]), flag=True)


folder="../objs/"
#Figure 5, Figure 6
#plot_eu()

#Figure 7
#plot_uu()

#Figure 8, 9
#plot_uu_minus_eu_rdmp()



#temporal
plot_temporal()