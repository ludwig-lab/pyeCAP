# python standard library imports
import warnings

# Scientific computing imports
import numpy as np

# Matplotlib imports
from matplotlib import get_backend as mpl_get_backend
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _plt_setup_fig_axis(axis=None, fig_size=(5, 3), subplots=(1, 1), **kwargs):
    """Convenience function to setup a figure axis

    Parameters
    ----------
    axis : None, matplotlib.axis.Axis
        Either None to use a new axis or matplotlib axis to plot on.
    fig_size : tuple, list, np.ndarray
        The size (width, height) of the matplotlib figure.
    subplots : tuple
        The num rows, num columns of axis to plot on.
    **kwargs
        Keyword arguments to pass to fig.add_subplot.
    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib figure reference.
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    """
    plt.ioff()
    if axis is None:
        fig, ax = plt.subplots(*subplots, figsize=fig_size, **kwargs)
    else:
        fig = axis.figure
        ax = axis
    return fig, ax


def _plt_ax_to_pix(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = np.round(bbox.width*fig.dpi).astype(int)
    height = np.round(bbox.height*fig.dpi).astype(int)
    return width, height


def _plt_ax_aspect(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return float(bbox.height / bbox.width)


def _plt_check_interactive(show):
    backend = mpl_get_backend()
    if backend == 'module://ipympl.backend_nbagg':
        return show
    else:
        if show=='notebook':
            warnings.warn("Interactive backend required to plotting in 'notebook' mode. Use Jupyter Notebook magic "
                          "function '%matplotlib widget' to set matplotlib to use an interactive backend. Falling "
                          "back to using a non-interactive backend for plotting.")
            return True
        else:
            return show


def _plt_show_fig(fig, ax, show):
    """Convenience function to show a figure axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        matplotlib figure reference.
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    show: bool
        Boolean value indicating if the plot should be displayed.
    Returns
    -------
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    """
    if show:
        backend = mpl_get_backend()
        fig.tight_layout()
        if backend == 'module://ipympl.backend_nbagg':
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.show()
        else:
            plt.show()
            plt.close(fig)
        return None
    else:
        return ax


def _plt_add_cbar_axis(fig, ax, location='right', c_lim=None, c_label=None, c_map=None):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    divider = make_axes_locatable(ax)
    dax = divider.append_axes(location, size=0.2, pad=0.1)
    if c_map is not None:
        c_norm = mpl.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
        mpl.colorbar.ColorbarBase(dax, cmap=plt.get_cmap(c_map), norm=c_norm)
    if location == 'left':
        dax.tick_params(left=True, right=False, bottom=False, top=False,
                        labelleft=True, labelright=False, labelbottom=False, labeltop=False)
        dax.yaxis.set_label_position(location)
        dax.set_ylim(c_lim)
        dax.set_ylabel(c_label)
    elif location == 'right':
        dax.tick_params(left=False, right=True, bottom=False, top=False,
                        labelleft=False, labelright=True, labelbottom=False, labeltop=False)
        dax.yaxis.set_label_position(location)
        dax.set_ylim(c_lim)
        dax.set_ylabel(c_label)
    elif location == 'bottom':
        dax.tick_params(left=False, right=False, bottom=True, top=False,
                        labelleft=False, labelright=False, labelbottom=True, labeltop=False)
        dax.xaxis.set_label_position(location)
        dax.set_xlim(c_lim)
        dax.set_xlabel(c_label)
    elif location == 'top':
        dax.tick_params(left=False, right=False, bottom=False, top=True,
                        labelleft=False, labelright=False, labelbottom=False, labeltop=True)
        dax.xaxis.set_label_position(location)
        dax.set_xlim(c_lim)
        dax.set_xlabel(c_label)
    else:
        raise ValueError('Unrecognized colorbar location')
    return dax


def _plt_add_ax_connected_top(fig, ax, ratio=0.1):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    divider = make_axes_locatable(ax)
    dax = divider.append_axes("top", size=bbox.height*ratio, pad=0, sharex=ax)
    dax.get_xaxis().set_visible(False)
    return dax
