from matplotlib import rcParams
from matplotlib import pyplot as plt

from cycler import cycler

rcParams['font.family'] = 'Exo 2'
colors = [ "#009FE3", "#7DB928","#162448","#FFB500"]

plt.rcParams["axes.prop_cycle"] =cycler(color=colors)

def style_axes(*axs) -> None:
    for ax in axs:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
    ax.grid(True, which='major', axis='both', alpha=0.3)
    