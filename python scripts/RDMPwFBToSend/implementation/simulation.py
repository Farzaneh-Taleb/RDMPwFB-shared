"""
@author Farzaneh.Tlb
5/27/21 9:53 PM
Implementation of .... (Fill this line)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import Extended_RDMP as RDMP

folder="../results/"
folder2="../objs/"
p = .3
q = .4
d = .8
m = 4
n_block = 1000
alpha = .1
# th = 0.3
matplotlib.rc('xtick', labelsize=35)
matplotlib.rc('ytick', labelsize=35)
# matplotlib.rc('ylabel', labelsize=40)
# matplotlib.rc('ylabel', labelsize=40)
plt.rcParams['xtick.major.pad'] = '30'
plt.rcParams['ytick.major.pad'] = '30'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

y_loc = (0, 2, 4, 6, 8)
y_label = (r'$\bf{30}$', r'$\bf{70}$', r'$\bf{110}$', r'$\bf{150}$', r'$\bf{190}$')



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename,i,j):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        # print(obj)
    return obj[i][j]


def simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC, flagACC2BLA,
                     flagST2BLA, flagST, th=1., type=1):
    # time = l_block * n_block
    model_stable = RDMP.RRDMP_complete(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                       flagACC2BLA, flagST2BLA, flagST, th, type)

    model_stable = model_stable.run_trials()
    return model_stable


def plot_contour(p_range1, p_range2, l_block_range, values, name, vmin=None, vmax=None, title='', flag_diff=False):
    fig, ax0 = plt.subplots(figsize=(25, 20))

    cmap = plt.cm.get_cmap("jet", 256)
    levels = [0, 1000]
    cs1 = ax0.contourf(np.arange(0, p_range1.shape[0]), np.arange(0, l_block_range.shape[0]), values, levels, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    cs = ax0.contourf(np.arange(0, p_range1.shape[0]), np.arange(0, l_block_range.shape[0]), values, cmap=cmap,
                      vmin=vmin, vmax=vmax)

    matplotlib.rcParams.update({'font.family': 'Arial Narrow'})

    plt.yticks(y_loc, y_label, color='k', size=35)
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 8, 10, 12), (r'$\bf{0.2/0.1}$', r'$\bf{0.4/0.2}$',
                                                  r'$\bf{0.4/0.1}$', r'$\bf{0.6/0.2}$',
                                                  r'$\bf{0.9/0.3}$', r'$\bf{0.9/0.6}$',
                                                  r'$\bf{0.6/0.4}$', r'$\bf{0.7/0.3}$',
                                                  r'$\bf{0.8/0.2}$', r'$\bf{0.9/0.1}$'), color='k', size=27)
    plt.tick_params(axis='both', which='major', pad=30)
    plt.xlabel(r'$\bf{uncertain  \longleftarrow p_{R}(B)/p_{R}(w)  \longrightarrow  certain}$', color='k', size=35)
    plt.ylabel(r'$\bf{Volatie  \longleftarrow L  \longrightarrow  Stable}$', color='k', size=35)
    ax0.set_title(title, size=40, fontweight='bold')
    cs = plt.cm.ScalarMappable(cmap=cmap)
    cs.set_array(values)
    cs.set_clim(vmin, vmax)
    if (not flag_diff):
        cbar = fig.colorbar(cs, ax=ax0, ticks=[0.3, 0.5, 0.7, 0.9])
        cbar.ax.set_yticklabels(
            [r'$\bf{0.3}$', r'$\bf{0.5}$', r'$\bf{0.7}$', r'$\bf{0.9}$'], size=35)  # vertically oriented colorbar
    else:
        cbar = fig.colorbar(cs, ax=ax0)

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(folder+ name + '.pdf')
    plt.savefig(folder+ name + '.jpg')
    plt.close(fig)


def plot_heatmap(p1_range, p2_range, l_block_range, values, name, vmin=None, vmax=None, title='', flag_diff=False):
    fig, ax0 = plt.subplots(figsize=(40, 20))
    cmap = plt.cm.get_cmap("jet", 256)
    p_range1 = p1_range.tolist()
    p_range2 = p2_range.tolist()
    p_range1 = ['{:.1f}'.format(x) + "/" + '{:.1f}'.format(x) for (x, y) in zip(p_range1, p_range2)]
    l_range = l_block_range.tolist()
    ax0.set_xticks(np.arange(values.shape[1]), minor=False)
    ax0.set_yticks(np.arange(values.shape[0]), minor=False)
    ax0.set_xticklabels(p_range1, minor=False)
    ax0.set_yticklabels(l_range, minor=False)
    plt.yticks(y_loc, y_label, color='k', size=35)
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 8, 10, 12), (r'$\bf{0.2/0.1}$', r'$\bf{0.4/0.2}$',
                                                  r'$\bf{0.4/0.1}$', r'$\bf{0.6/0.2}$',
                                                  r'$\bf{0.9/0.3}$', r'$\bf{0.9/0.6}$',
                                                  r'$\bf{0.6/0.4}$', r'$\bf{0.7/0.3}$',
                                                  r'$\bf{0.8/0.2}$', r'$\bf{0.9/0.1}$'), color='k', size=27,
               rotation=90)

    plt.xlabel(r'$\bf{uncertain  \longleftarrow p_{R}(B)/p_{R}(w)  \longrightarrow  certain}$', color='k', size=35,
               loc='right')
    plt.ylabel(r'$\bf{Volatie  \longleftarrow L  \longrightarrow  Stable}$', color='k', size=35)
    im = ax0.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.vlines(5.5, -0.5, 7.5, linestyles="--", color='red', linewidth=10)
    ax0.invert_yaxis()
    ax0.set_title(title, size=30, fontweight='bold')
    cs = plt.cm.ScalarMappable(cmap=cmap)
    cs.set_array(values)
    cs.set_clim(vmin, vmax)
    if (not flag_diff):
        cbar = fig.colorbar(cs, ax=ax0, ticks=[0.3, 0.5, 0.7, 0.9])
        cbar.ax.set_yticklabels(
            [r'$\bf{0.3}$', r'$\bf{0.5}$', r'$\bf{0.7}$', r'$\bf{0.9}$'], size=35)  # vertically oriented colorbar
    else:
        cbar = fig.colorbar(cs, ax=ax0)

    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(folder+ name + '.jpg')
    plt.close(fig)


def plot_contour_heatmap(p1_range, p2_range, l_block_range, values, name, vmin=None, vmax=None, title='',
                         flag_diff=False):
    fig, ax0 = plt.subplots(1, 2, figsize=(40, 20))

    cmap = plt.cm.get_cmap("jet", 256)

    p_range1 = p1_range[:6].tolist()
    p_range2 = p2_range[:6].tolist()
    p_range1 = ['{:.1f}'.format(x) + "/" + '{:.1f}'.format(y) for (x, y) in zip(p_range1, p_range2)]
    l_range = l_block_range.tolist()
    print(values.shape[1])
    print(values.shape[0], values.shape[1])
    ax0[0].set_xticks(np.arange(6), minor=False)
    ax0[0].set_yticks(np.arange(values.shape[0]), minor=False)
    ax0[0].set_xticklabels(p_range1, minor=False)
    ax0[0].set_yticklabels(l_range, minor=False)

    plt.yticks(y_loc, y_label, color='k',
               size=35)
    plt.xticks((0, 1, 2, 3, 4, 5), (r'$\bf{0.2/0.1}$', r'$\bf{0.4/0.2}$',
                                    r'$\bf{0.4/0.1}$', r'$\bf{0.6/0.2}$',
                                    r'$\bf{0.9/0.3}$', r'$\bf{0.9/0.6}$',
                                    ), color='k', size=35)

    im = ax0[0].imshow(values[:, :6], cmap=cmap, vmin=vmin, vmax=vmax)
    ax0[0].invert_yaxis()
    cmap = plt.cm.get_cmap("jet", 256)
    levels = [0, 1000]

    print(values.shape)
    cs1 = ax0[1].contourf(np.arange(0, 7), np.arange(0, l_block_range.shape[0]), values[:, 6:], levels, cmap=cmap,
                          vmin=vmin, vmax=vmax)
    cs = ax0[1].contourf(np.arange(0, 7), np.arange(0, l_block_range.shape[0]), values[:, 6:], cmap=cmap, vmin=vmin,
                         vmax=vmax)

    matplotlib.rcParams.update({'font.family': 'Arial Narrow'})
    plt.xticks((0, 2, 4, 6), (r'$\bf{0.6/0.4}$', r'$\bf{0.7/0.3}$',
                              r'$\bf{0.8/0.2}$', r'$\bf{0.9/0.1}$'), color='k', size=27)
    plt.xlabel(r'$\bf{uncertain  \longleftarrow p_{R}(B)/p_{R}(w)  \longrightarrow  certain}$', color='k', size=35)
    cs = plt.cm.ScalarMappable(cmap=cmap)
    cs.set_array(values)
    cs.set_clim(vmin, vmax)
    if (not flag_diff):
        cbar = fig.colorbar(cs, ax=ax0, ticks=[0.3, 0.5, 0.7, 0.9])
        cbar.ax.set_yticklabels(
            [r'$\bf{0.3}$', r'$\bf{0.5}$', r'$\bf{0.7}$', r'$\bf{0.9}$'], size=35)  # vertically oriented colorbar
    else:
        cbar = fig.colorbar(cs, ax=ax0)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='major', pad=30)
    fig.supylabel(r'$\bf{Volatie  \longleftarrow L  \longrightarrow  Stable}$', color='k', size=35)
    fig.suptitle(title, color='k', size=35, fontweight='bold')
    plt.savefig(folder + name + '.pdf')
    plt.savefig(folder + name + '.jpg')
    plt.close(fig)


def plot_heatmap_separated(p1_range, p2_range, l_block_range, values, name, vmin=None, vmax=None, title='',
                           flag_diff=False):
    values1 = values[:, :6]
    values2 = values[:, 6:]
    fig, ax0 = plt.subplots(1, 2, figsize=(40, 20), sharey=True)
    cmap = plt.cm.get_cmap("jet", 256)
    p_range1 = p1_range.tolist()
    p_range2 = p2_range.tolist()
    p_range1 = ['{:.1f}'.format(x) + "/" + '{:.1f}'.format(x) for (x, y) in zip(p_range1, p_range2)]
    l_range = l_block_range.tolist()

    ax0[0].set_yticks(y_loc, minor=False)
    ax0[0].set_yticklabels(y_label, minor=False)

    ax0[0].set_xticks(np.arange(values1.shape[1]), minor=False)
    ax0[0].set_xticklabels((r'$\bf{0.2/0.1}$', r'$\bf{0.3/0.1}$',
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

    fig.subplots_adjust(wspace=0.1, left=0.02, bottom=0.2, top=0.9)
    ax0[0].tick_params(which='both', width=5)
    ax0[0].tick_params(which='major', length=14)
    ax0[1].tick_params(which='both', width=5)
    ax0[1].tick_params(which='major', length=14)
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
    plt.savefig(folder+ name + '.jpg')
    plt.savefig(folder+ name + '.pdf')
    plt.close(fig)


def plot_all(p_range1, p_range2, l_block_range, values, name, vmin, vmax, title, flag_diff):
    plot_heatmap_separated(p_range1, p_range2, l_block_range, values, "SepHeatmap_" + name, vmin, vmax, title,
                           flag_diff)
    np.savetxt(folder + "_" + name + ".csv", values, delimiter=",")


def exp_unc():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 110, 10)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            model = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                     flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            items[i, j] = generateMean(model.p_acc[1], l_block)

    plot_contour(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp", vmin=0.3, vmax=0.9,
                 title='Performance of Full Model', flag_diff=False)
    plot_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp_heat", vmin=0.3, vmax=0.9,
                 title='Performance of Full Model', flag_diff=False)
    plot_heatmap_separated(p_range1, p_range2, l_block_range, np.transpose(items), "performance_heat_sep", vmin=0.3,
                           vmax=0.9, title='Performance of Full Model', flag_diff=False)
    plot_contour_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp_heat_count",
                         vmin=0.3, vmax=0.9, title='Performance of Full Model', flag_diff=False)


def generateOverTime(values, l_max):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    item_mean_padded = np.pad(item_mean, (0, l_max - item_mean.shape[0]), 'constant', constant_values=(0, 0))
    return item_mean_padded


def generateMean(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    item_mean_padded = np.mean(item_mean[:l_block])
    return item_mean_padded

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

def generateMeanBeforeRev2(values, l_block):
    #
    item_mean = np.mean(np.asarray(np.split(values, n_block / 2)), 0)
    # item_mean_padded = np.mean(item_mean[l_block-int(l_block/5):l_block+1])
    item_mean_padded = np.mean(item_mean[l_block-4*int(l_block/5)+1:l_block])
    return item_mean_padded


def performance_rdmp():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 110, 10)

    flagBLA2ACC = False
    flagACC2BLA = False
    flagST2BLA = False
    flagST = True
    type_decision = 2

    # items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],  2 * int(np.amax(l_block_range))])
    items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                     flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            items[i, j] = generateMean(model.p_acc[1], l_block)

    # print(p_range1[1:-1])
    # p=.4,l=110
    # items[0,-1]=1
    # items[1,3]=1
    # items[5,1]=0
    # items[6,3]=1
    # items[11,1]=0
    plot_contour(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp", vmin=0.3, vmax=0.9,
                 title='Performance of RDMP Model', flag_diff=False)
    plot_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp_heat", vmin=0.3, vmax=0.9,
                 title='Performance of RDMP Model', flag_diff=False)
    plot_heatmap_separated(p_range1, p_range2, l_block_range, np.transpose(items), "performance_heat_sep", vmin=0.3,
                           vmax=0.9, title='Performance of RDMP Model', flag_diff=False)
    plot_contour_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_rdmp_heat_count",
                         vmin=0.3, vmax=0.9, title='Performance of RDMP Model', flag_diff=False)


def performance_full():
    # p_range1 = np.asarray([0.65, 0.75, 0.85])
    # p_range2 = np.asarray([0.35, 0.25, 0.15])
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 110, 10)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    # items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],  2 * int(np.amax(l_block_range))])
    items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                     flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            items[i, j] = generateMean(model.p_acc[1], l_block)

    plot_contour(p_range1, p_range2, l_block_range, np.transpose(items), "performance_full", vmin=0.3, vmax=0.9,
                 title='Performance of Full Model', flag_diff=False)
    plot_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_full_heat", vmin=0.3, vmax=0.9,
                 title='Performance of Full Model', flag_diff=False)
    plot_heatmap_separated(p_range1, p_range2, l_block_range, np.transpose(items), "performance_full_sep", vmin=0.3,
                           vmax=0.9, title='Performance of Full Model', flag_diff=False)
    plot_contour_heatmap(p_range1, p_range2, l_block_range, np.transpose(items), "performance_full_heat_count",
                         vmin=0.3, vmax=0.9, title='Performance of Full Model', flag_diff=False)


def diffperformance_full_plastic():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    p_dest = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)
            p_dest[i, j] = generateMean(model_full.destablization_count, l_block)

    flagBLA2ACC = False
    flagACC2BLA = False
    flagST2BLA = False
    flagST = True
    type_decision = 1

    # items = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0],  2 * int(np.amax(l_block_range))])
    performance_plastic = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_plastic[i, j] = generateMean(model_full.p_st[1], l_block)
    dif_full_plastic = perofmance_full - performance_plastic
    plot_all(p_range1, p_range2, l_block_range, np.transpose(p_dest), "probability_dest", vmin=None, vmax=None,
             title='Probability of Destabilization', flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_plastic), "performance_plastic", vmin=0.5,
             vmax=0.9,
             title='Performance of Plastic Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_plastic), "performance_full_plastic", vmin=None,
             vmax=None, title='Differnece Between Performance of the Full Model and Plastic Model', flag_diff=True)

def diffperformance_full_staticNotMeanRemoved():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2
    th = 1

    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = False
    type_decision = 2

    th = 0.1
    performance_static = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_static[i, j] = generateMean(model_full.p_acc[1], l_block)
            # p_dest[i, j] = generateMean(model_full.destablization_count, l_block)
    dif_full_plastic = perofmance_full - performance_static
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_static), "performance_static" + str(th),
             vmin=0.5, vmax=0.9,
             title='Performance of Model with Static Threshold=' + str(th), flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_plastic), "performance_full_static" + str(th),
             vmin=None, vmax=None,
             title='Differnece Between Performance of the Full Model and Model with Static Threshold=' + str(th),
             flag_diff=True)

    th = 0.05
    performance_static = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_static[i, j] = generateMean(model_full.p_acc[1], l_block)
    dif_full_plastic = perofmance_full - performance_static
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_static), "performance_static" + str(th),
             vmin=0.5, vmax=0.9,
             title='Performance of Model with Static Threshold=' + str(th), flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_plastic), "performance_full_static" + str(th),
             vmin=None, vmax=None,
             title='Differnece Between Performance of the Full Model and Model with Static Threshold=' + str(th),
             flag_diff=True)

    th = 0.15
    performance_static = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_static[i, j] = generateMean(model_full.p_acc[1], l_block)

    dif_full_plastic = perofmance_full - performance_static
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_static), "performance_static" + str(th),
             vmin=0.5, vmax=0.9,
             title='Performance of Model with Static Threshold=' + str(th), flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_plastic), "performance_full_static" + str(th),
             vmin=None, vmax=None,
             title='Differnece Between Performance of the Full Model and Model with Static Threshold=' + str(th),
             flag_diff=True)

    th = 0.2
    performance_static = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_static[i, j] = generateMean(model_full.p_acc[1], l_block)
    dif_full_plastic = perofmance_full - performance_static
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_static), "performance_static" + str(th),
             vmin=0.5, vmax=0.9,
             title='Performance of Model with Static Threshold=' + str(th), flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_plastic), "performance_full_static" + str(th),
             vmin=None, vmax=None,
             title='Differnece Between Performance of the Full Model and Model with Static Threshold=' + str(th),
             flag_diff=True)


def diffperformance_full_rdmp():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    p_dest = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4
            th=0 #it doesn't matter here
            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)
            p_dest[i, j] = generateMean(model_full.destablization_count, l_block)

    flagBLA2ACC = False
    flagACC2BLA = False
    flagST2BLA = False
    flagST = True
    type_decision = 2

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)

    dif_full_rdmp = np.subtract(perofmance_full, performance_rdmp)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(p_dest), "probability_dest", vmin=None, vmax=None,
             title='Probability of Destabilization', flag_diff=False)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_rdmp), "performance_rdmp", vmin=0.5, vmax=0.9,
             title='Performance of RDMP Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_rdmp), "performance_full_rdmp", vmin=None,
             vmax=None, title='Differnece Between Performance of the Full Model and RDMP Model', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(ex_un_rdmp), "eu_rdmp", vmin=None, vmax=None,
             title='Average of Expected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(unex_un_rdmp), "un_rdmp", vmin=None, vmax=None,
             title='Average of Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(ex_un_meanRemoved_rdmp), "eu_meanRemov_rdmp", vmin=None,
             vmax=None, title='Average of Expected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(unex_un_meanRemoved_rdmp), "un_meanRemov_rdmp", vmin=None,
             vmax=None, title='Average of Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_meanRemov_rdmp), "dif_meanRemov_rdmp", vmin=None,
             vmax=None, title='Difference Between Expected and Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_rdmp), "dif_rdmp", vmin=None, vmax=None,
             title='Difference Between Expected and Unexpected Uncertainty', flag_diff=True)


def signals_rdmp():

    p_range1 = np.asarray([0.9, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    # 13
    # 8
    l_block_range = np.arange(30,200,20)


    flagBLA2ACC = False
    flagACC2BLA = False
    flagST2BLA = False
    flagST = True
    type_decision = 2

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    eu_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    eu_meta_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    uu_before2 = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])


    # models = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    # Creates a list containing 5 lists, each of 8 items, all set to 0
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]
    # print(models)

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            # p=.3
            # q=.4
            th=0
            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)

            eu[i, j] = generateMean(model_rdmp.rpe, l_block)
            eu_meta[i, j] = generateMean(model_rdmp.rpe_meta, l_block)


            eu_before2[i, j] = generateMeanBeforeRev2(model_rdmp.rpe, l_block)
            eu_meta_before2[i, j] = generateMeanBeforeRev2(model_rdmp.rpe_meta, l_block)

            eu_after[i, j] = generateMeanAfterRev(model_rdmp.rpe, l_block)
            eu_meta_after[i, j] = generateMeanAfterRev(model_rdmp.rpe_meta, l_block)

            uu_before2[i, j] = generateMeanBeforeRev2(np.subtract(model_rdmp.deltaF_s ,model_rdmp.deltaF_w), l_block)




            models[i][j] = model_rdmp

    # Figure 5
    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_before), "eu_all_before_",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Plastic Synapses before Reversal ', flag_diff=True)

    # Figure 6
    plot_all(p_range1, p_range2, l_block_range, np.transpose(eu_meta_before), "eu_meta_all_before",
             # vmin=min(np.min(eu_before), np.min(eu_meta_before)),
             # vmax=max(np.max(eu_before), np.max(eu_meta_before)),
             vmin=.1, vmax=.6,
             title='Expected Uncertainty from Meta-plastic Synapses before Reversal', flag_diff=True)


    #Figure 7
    plot_all(p_range1, p_range2, l_block_range, np.transpose(uu_before2), "uu_before2",
             # vmin=np.min(uu_after), vmax=np.max(uu_after),
             vmin=.1, vmax=.24,
             title='Unexpected Uncertainty from Meta-plastic Synapses brefore Reversal' , flag_diff=True)


    save_object(models, '../objs/'+'modelll_rdmp2_1000.pkl')


def signals_full():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30,200,20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2
    th=1

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_full_before = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    performance_full_after = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            performance_full_before[i, j] = generateMeanBeforeRev2(model_rdmp.destablization_count, l_block)
            performance_full_after[i, j] = generateMeanAfterRev(model_rdmp.destablization_count, l_block)
            models[i][j] = model_rdmp


    #Figure 8
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_before), "rdmpWf_dest_before",
             # vmin=0, vmax=.8,
             vmin=0, vmax=.8,
             title='Probability of Destabilization', flag_diff=True)
    #Figure 9
    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_full_after), "rdmpWf_dest_after",
             # vmin=0, vmax=.8,
             vmin=0, vmax=.8,
             title='Probability of Destabilization', flag_diff=True)





    save_object(models, folder2+'modelll_full2_1000.pkl')


def signals_random():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30,200,20)


    flagBLA2ACC = False
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp


    save_object(models, '../objs/'+'modelll_random2_1000.pkl')
def signals_plastic():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30,200,20)

    flagBLA2ACC = False
    flagACC2BLA = False
    flagST2BLA = False
    flagST = False
    type_decision = 1

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_plastic_1000.pkl')


def signals_static():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30,200,20)


    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = False
    flagST = True
    type_decision = 2

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0.02, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_staticth0.02_1000.pkl')

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0.01, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_staticth0.01_1000.pkl')

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_staticth0_1000.pkl')
def signals_euOnly():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30,200,20)
    flagBLA2ACC = True
    flagACC2BLA = False
    flagST2BLA = True
    flagST = True
    type_decision = 2

    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    w, h = l_block_range.shape[0], p_range1.shape[0]
    models = [[0 for x in range(w)] for y in range(h)]


    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0.1, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_euOnlyth0.1_1000.pkl')

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, 0.4, type_decision)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)
            models[i][j] = model_rdmp
    save_object(models, '../objs/'+'modelll_euOnlyth0.4_1000.pkl')


def plot_signals_rdmp():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)
    performance_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    ex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    unex_un_meanRemoved_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_meanRemov_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    dif_rdmp = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_rdmp = load_object("model_rdmp_1000.pkl", i, j)
            performance_rdmp[i, j] = generateMean(model_rdmp.p_acc[1], l_block)
            ex_un_rdmp[i, j] = generateMean(model_rdmp.rpe_avg, l_block)
            unex_un_rdmp[i, j] = generateMean(model_rdmp.deltaF_F, l_block)
            ex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.rpe_avg_meanRmv, l_block)
            unex_un_meanRemoved_rdmp[i, j] = generateMean(model_rdmp.deltaF, l_block)
            dif_meanRemov_rdmp[i, j] = generateMean(model_rdmp.dif_meanRemov, l_block)
            dif_rdmp[i, j] = generateMean(model_rdmp.dif, l_block)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_rdmp), "performance_rdmp", vmin=0.5, vmax=0.9,
             title='Performance of RDMP Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(ex_un_rdmp), "eu_rdmp", vmin=None, vmax=None,
             title='Average of Expected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(unex_un_rdmp), "un_rdmp", vmin=None, vmax=None,
             title='Average of Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(ex_un_meanRemoved_rdmp), "eu_meanRemov_rdmp", vmin=None,
             vmax=None, title='Average of Expected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(unex_un_meanRemoved_rdmp), "un_meanRemov_rdmp", vmin=None,
             vmax=None, title='Average of Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_meanRemov_rdmp), "dif_meanRemov_rdmp", vmin=None,
             vmax=None, title='Difference Between Expected and Unexpected Uncertainty', flag_diff=True)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_rdmp), "dif_rdmp", vmin=None, vmax=None,
             title='Difference Between Expected and Unexpected Uncertainty', flag_diff=True)


def diffperformance_full_random():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)

    flagBLA2ACC = False
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_static = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                            flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_random[i, j] = generateMean(model_static.p_acc[1], l_block)

    dif_full_random = np.subtract(perofmance_full, performance_random)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random), "performance_random", vmin=0.5,
             vmax=0.9,
             title='Performance of Model with Random Destabilization', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full3", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_random), "performance_full_random", vmin=None,
             vmax=None, title='Differnece Between Performance of the Full Model and Model with Random Destabilization',
             flag_diff=True)


def diffperformance_full_eu():
    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])
    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)

    flagBLA2ACC = True
    flagACC2BLA = False
    flagST2BLA = True
    flagST = True
    type_decision = 2

    performance_random = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)

            model_static = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                            flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_random[i, j] = generateMean(model_static.p_acc[1], l_block)

    dif_full_random = np.subtract(perofmance_full, performance_random)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_random), "performance_eu", vmin=0.5, vmax=0.9,
             title='Performance of Model with ACC to BLA Lesion', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full4", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_random), "performance_full_eu", vmin=None,
             vmax=None, title='Differnece Between Performance of the Full Model and Model with Random Destabilization',
             flag_diff=True)


def diffperformance_full_static():

    p_range1 = np.asarray([0.2, .3, .4, .6, .9, .9, .6, .65, .7, .75, .8, .85, .9])
    p_range2 = np.asarray([0.1, .1, .2, .2, .3, .6, .4, .35, .3, .25, .2, .15, .1])

    l_block_range = np.arange(30, 200, 20)

    flagBLA2ACC = True
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2
    perofmance_full = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])

    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            model_full = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                          flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            perofmance_full[i, j] = generateMean(model_full.p_acc[1], l_block)

    # Random
    flagBLA2ACC = False
    flagACC2BLA = True
    flagST2BLA = True
    flagST = True
    type_decision = 2

    performance_static = np.zeros(shape=[p_range1.shape[0], l_block_range.shape[0]])
    for i in range(p_range1.shape[0]):
        for j, l_block in enumerate(l_block_range):
            l_block = int(l_block)
            prob1 = p_range1[i]
            prob2 = p_range2[i]
            print(i, j)
            model_static = simulate_by_type(p, q, d, m, prob1, prob2, l_block, n_block, alpha, flagBLA2ACC,
                                            flagACC2BLA, flagST2BLA, flagST, th, type_decision)
            performance_static[i, j] = generateMean(model_static.p_acc[1], l_block)

    dif_full_rdmp = np.subtract(perofmance_full, performance_static)

    plot_all(p_range1, p_range2, l_block_range, np.transpose(performance_static), "performance_static_" + str(th),
             vmin=0.5, vmax=0.9,
             title='Performance of Model with Striatum Lesion', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(perofmance_full), "performance_full2", vmin=0.5, vmax=0.9,
             title='Performance of Full Model', flag_diff=False)
    plot_all(p_range1, p_range2, l_block_range, np.transpose(dif_full_rdmp), "performance_full_static_" + str(th),
             vmin=None, vmax=None,
             title='Differnece Between Performance of the Full Model and Model with Striatum Lesion', flag_diff=True)


signals_rdmp()
#performance_full()
# diffperformance_full_random()
# diffperformance_full_eu()

#performance_rdmp()
# performance_full()
#diffperformance_full_rdmp()
# diffperformance_full_plastic()
# diffperformance_full_staticNotMeanRemoved()
# diffperformance_full_static()
#
#signals_rdmp()
# signals_euOnly()
#signals_full()

# signals_random()
# signals_plastic()
# signals_static()
# plot_signals_rdmp()