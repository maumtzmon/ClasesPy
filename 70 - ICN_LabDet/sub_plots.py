import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['figure.figsize'] = 8, 6
# plt.rcParams['figure.max_open_warning'] = 50

nameList=[['x1','y1'],['x2','y1'],['x1','y2'],['x2','y2']]
titleList=['Title_1', 'Title_2','Title_3', 'Title_4']
test_data = [[1,2],[2,1],[1,1],[1,2]]


def example_plot(ax, data, label, title, fontsize=8, hide_labels=False):
    ax.plot(data) #data to plot

    ax.locator_params(nbins=5) #how many values on x and y axis
    ax.set_xlabel(label[0], fontsize=fontsize)
    ax.set_ylabel(label[1], fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

def main(data=test_data):
    canvas2plot, axs = plt.subplots(2, 2, constrained_layout=True)

    for canvas2plot,data,label, title in zip(axs.flat, test_data, titleList, nameList):
        example_plot(canvas2plot, data, label,title)

    plt.show()

if __name__== "__main__":
    exitcode= main()
    exit(code=exitcode)