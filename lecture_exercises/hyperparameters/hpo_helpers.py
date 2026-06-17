import matplotlib.pyplot as plt
def decorate_plot(ax, xlabel=None, ylabel=None, legend=False):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    if legend:
        ax.legend(frameon=False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.set_xlim(0,20)

def double_plot(ax, xx, yy, col, label, ll=None, aa=1):
    # Do the plotting twice, because we only want to add to the legend once
    ax.plot(xx, yy.T.iloc[0].T, c=col, linestyle=ll, alpha=aa,
                label=label);
    ax.plot(xx, yy, c=col, linestyle=ll, alpha=aa,);