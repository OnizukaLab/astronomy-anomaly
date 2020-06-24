import matplotlib.pyplot as plt


def set_rcParams() -> None:
    """
    plt.rcParams を設定する。
    """
    
    # plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.4
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1
