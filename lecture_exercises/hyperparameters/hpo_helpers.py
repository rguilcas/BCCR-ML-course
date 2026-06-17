def decorate_plot(ax, xlabel=None, ylabel=None, legend=False):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    if legend:
        ax.legend(frameon=False)

def double_plot(ax, xx, yy, col, label, ll=None, aa=1):
    # Do the plotting twice, because we only want to add to the legend once
    ax.plot(xx, yy.T.iloc[0].T, c=col, linestyle=ll, alpha=aa,
                label=label);
    ax.plot(xx, yy, c=col, linestyle=ll, alpha=aa,);