import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, OrderedDict
import os
import numpy as np
import pickle


def scatter_plot(name_array, y_array, x_array, title, xlabel, ylabel, xlim_min=None, x_lim_max=None,
                 ylim_min=None, ylim_max=None, var_array=None):
    assert(len(name_array)==len(y_array))
    char_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--','-.',':','*',',','+']
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111)
    for i in range(len(y_array)):
        ax.plot(x_array[i], y_array[i], char_colors[i]+linestyles[i], label=name_array[i], linewidth=2, markersize=7)
        if var_array is not None:
            plt.fill_between(x_array[i], y_array[i] - var_array[i], y_array[i] + var_array[i], alpha=.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle='--')
    plt.xlim([xlim_min, x_lim_max])
    plt.ylim([ylim_min, ylim_max])
    box = ax.get_position()
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


def remove_first_instance(it_arr):
    for i in range(it_arr):
        first_idx = np.where(it_arr[i].arr[0])[1]
        for j in range(len(it_arr[i].arr) - 1):
            for idx in first_idx:
                (it_arr[i].arr[j + 1])[idx] = False
    return it_arr


def get_intersection_percent(array_paths):  # TODO: Verify and debug this function please!
    it_arr = []
    for i in range(len(array_paths)):
        with open(array_paths[i], 'rb') as handle:
            it_arr.append(pickle.load(handle))

    it_arr = remove_first_instance(it_arr)

    detailed = True
    for i in range(len(it_arr) - 1):
        if len(it_arr[i].arr) != len(it_arr[i + 1].arr):
            detailed = False
            break

    if detailed:
        print('Detailed Results')
        for j in range(len(it_arr[0].arr)):
            num_points = it_arr[0].arr[j].sum()
            intersect_arr = np.logical_and(it_arr[0].arr[j], it_arr[1].arr[j])
            for i in range(len(it_arr) - 2):
                intersect_arr = np.logical_and(intersect_arr, it_arr[i + 2].arr[j])
                assert (it_arr[i].arr[j].sum() == num_points)
            intersect_percent = (intersect_arr.sum() * 1.0) / (num_points * 1.0)
            print(f"Acq Itr: {j + 1}/{len(it_arr[0].arr)}: {intersect_percent}")

    num_points = it_arr[0].arr[-1].sum()
    intersect_arr = np.logical_and(it_arr[0].arr[-1], it_arr[1].arr[-1])
    for i in range(len(it_arr) - 2):
        intersect_arr = np.logical_and(intersect_arr, it_arr[i + 2].arr[-1])
        assert (it_arr[i].arr[-1].sum() == num_points)
    intersect_percent = (intersect_arr.sum() * 1.0) / (num_points * 1.0)
    print(f"Overall intersection: {intersect_percent}")


def get_intersection_across_seeds(expname, num_seeds):
    sp = expname.split('+')
    exps = []
    for i in range(num_seeds):
        sp[-4] = str(i)
        exps.append("".join([item + '+' for item in sp[:-1]]))