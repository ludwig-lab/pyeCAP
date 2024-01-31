# python standard library imports
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

# Scientific computing imports
import numpy as np

# Matplotlib imports
from matplotlib import get_backend as mpl_get_backend
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
    width = np.round(bbox.width * fig.dpi).astype(int)
    height = np.round(bbox.height * fig.dpi).astype(int)
    return width, height


def _plt_ax_aspect(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return float(bbox.height / bbox.width)


def _plt_check_interactive(show):
    backend = mpl_get_backend()
    if backend == "module://ipympl.backend_nbagg":
        return show
    else:
        if show == "notebook":
            warnings.warn(
                "Interactive backend required to plotting in 'notebook' mode. Use Jupyter Notebook magic "
                "function '%matplotlib widget' to set matplotlib to use an interactive backend. Falling "
                "back to using a non-interactive backend for plotting."
            )
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
        if backend == "module://ipympl.backend_nbagg":
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.show()
        else:
            plt.show()
            plt.close(fig)
        return None
    else:
        return ax


def _plt_add_cbar_axis(fig, ax, location="right", c_lim=None, c_label=None, c_map=None):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    divider = make_axes_locatable(ax)
    dax = divider.append_axes(location, size=0.2, pad=0.1)
    if c_map is not None:
        c_norm = mpl.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
        mpl.colorbar.ColorbarBase(dax, cmap=plt.get_cmap(c_map), norm=c_norm)
    if location == "left":
        dax.tick_params(
            left=True,
            right=False,
            bottom=False,
            top=False,
            labelleft=True,
            labelright=False,
            labelbottom=False,
            labeltop=False,
        )
        dax.yaxis.set_label_position(location)
        dax.set_ylim(c_lim)
        dax.set_ylabel(c_label)
    elif location == "right":
        dax.tick_params(
            left=False,
            right=True,
            bottom=False,
            top=False,
            labelleft=False,
            labelright=True,
            labelbottom=False,
            labeltop=False,
        )
        dax.yaxis.set_label_position(location)
        dax.set_ylim(c_lim)
        dax.set_ylabel(c_label)
    elif location == "bottom":
        dax.tick_params(
            left=False,
            right=False,
            bottom=True,
            top=False,
            labelleft=False,
            labelright=False,
            labelbottom=True,
            labeltop=False,
        )
        dax.xaxis.set_label_position(location)
        dax.set_xlim(c_lim)
        dax.set_xlabel(c_label)
    elif location == "top":
        dax.tick_params(
            left=False,
            right=False,
            bottom=False,
            top=True,
            labelleft=False,
            labelright=False,
            labelbottom=False,
            labeltop=True,
        )
        dax.xaxis.set_label_position(location)
        dax.set_xlim(c_lim)
        dax.set_xlabel(c_label)
    else:
        raise ValueError("Unrecognized colorbar location")
    return dax


def _plt_add_ax_connected_top(fig, ax, ratio=0.1):
    """
    Add a new axis on top of the given axis, with a specified height ratio.

    Parameters:
    - fig: The matplotlib figure object
    - ax: The existing axis object
    - ratio: The height ratio of the new axis to the existing axis (default 0.1)

    Returns:
    - dax: The new axis object
    """

    # Get the bounding box of the existing axis in figure coordinates
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # Create an axes locator for the existing axis
    divider = make_axes_locatable(ax)

    # Get shared y-axes
    shared_y_axes = ax.get_shared_x_axes()

    # Check if there are any shared y-axes with overlapping coordinates
    if shared_y_axes:
        print([x for xs in shared_y_axes for x in xs])
        for shared_ax in [x for xs in shared_y_axes for x in xs]:
            print(shared_ax)
            if shared_ax.get_xlim() == ax.get_xlim():
                # Get the bounding box of the shared y-axis
                shared_bbox = shared_ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )

                # Check if the bounding boxes overlap
                if bbox.overlaps(shared_bbox):
                    # Split the shared y-axis
                    shared_ax_divider = make_axes_locatable(shared_ax)
                    shared_ax_dax = shared_ax_divider.append_axes(
                        "top", size=bbox.height * ratio, pad=0, sharex=shared_ax
                    )
                    shared_ax_dax.get_xaxis().set_visible(False)

    # Create a new axis on top of the existing axis
    dax = divider.append_axes("top", size=bbox.height * ratio, pad=0, sharex=ax)

    # Hide the x-axis of the new axis
    dax.get_xaxis().set_visible(False)

    # Return the new axis object
    return dax


def _plt_split_axis(ax, ratios, direction="horizontal"):
    """
    Split the given axis into multiple sub-axes with the specified ratios.

    Parameters:
    - ax: The axis to be split
    - ratios: A list of ratios, where each ratio is a float between 0 and 1
    - direction: The direction of the split, either 'horizontal' or 'vertical'

    Returns:
    - ax_list: A list of new axis objects
    """
    if direction not in ["horizontal", "vertical"]:
        raise ValueError("Direction must be either 'horizontal' or 'vertical'")

    # Normalize the ratios to ensure they add up to 1
    ratio_sum = sum(ratios)
    normalized_ratios = [ratio / ratio_sum for ratio in ratios]

    if ratio_sum != 1:
        warnings.warn(
            "Ratios do not add up to 1. Normalizing to fit available height.",
            UserWarning,
        )

    # Get the bounding box of the existing axis in figure coordinates
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())

    # Create an axes locator for the existing axis
    divider = make_axes_locatable(ax)

    # Create a list to store the new axis objects
    ax_list = []

    # Split the axis horizontally
    for i, ratio in enumerate(normalized_ratios):
        if direction == "horizontal":
            # Calculate the size of the new axis as a fraction of the original axis
            size = bbox.height * ratio

            # Create a new axis on top of the existing axis
            dax = divider.append_axes("top", size=size, pad=0, sharex=ax)

            # Hide the x-axis of the new axis
            dax.get_xaxis().set_visible(False)
        else:
            # Calculate the size of the new axis as a fraction of the original axis
            size = bbox.width * ratio

            # Create a new axis on top of the existing axis
            dax = divider.append_axes("right", size=size, pad=0, sharey=ax)

            # Hide the y-axis of the new axis
            dax.get_yaxis().set_visible(False)

        # Add the new axis to the list
        ax_list.append(dax)
    return ax_list
