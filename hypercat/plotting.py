from __future__ import print_function

__version__ = '20180215'   #yyymmdd
s__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Plotting funcs for hypercat.

.. automodule:: plotting
"""

# IMPORTS
# std lib
import numpy as N
from numpy import ma

# 3rd party
import pylab as p
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u

# own
from utils import arrayify

def plot_with_wcs(image):

    """Demonstration function to show plotting with WCS.

    Parameters
    ----------
    image : instance
        Instance of :class:`imageops.Image` class. Must have member
        ``.wcs`` which is an instance of :class:`astropy.wcs.wcs.WCS`

    Returns
    -------
    Nothing.

    Example
    -------
    Assuming you have an instance of :class:`imageops.Image`, e.g. ``sky``:

    .. code-block:: python

       import plotting
       plotting.plot_with_wcs(sky)
    """
    
    
    fig = p.figure(figsize=(6.5,6))
#    fig = p.figure()
    ax = fig.add_subplot(111, projection=image.wcs)
    xc = ax.coords[0]
    yc = ax.coords[1]
    ax.imshow(image.data.T, origin='lower', cmap=p.cm.viridis)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    xc.set_ticks(spacing=100.*u.mas)
    yc.set_ticks(spacing=100.*u.mas)

    fig.subplots_adjust(left=0.25,right=0.98,top=0.98,bottom=0.07)
    
#    ax.tick_params(
#    axis='both',          # changes apply to the x-axis
#    which='major',      # both major and minor ticks are affected
#    bottom='on',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    left='on',         # ticks along the top edge are off
#    right='off'         # ticks along the top edge are off
#    )
#    p.show()


def plotPanel(ax,image,units='',extent=None,colorbar=False,title='',cmap=p.cm.viridis,contours=None,interpolation='bicubic'):

    """Plot a single panel. To be called from :func:`multiplot()`
    """
    
    # what kind of animal is 'image'?
    cls = None
    try:
        cls = image.__class__.__name__
    except:
        raise

    # extract the data (2d array) to be plotted
    if cls in ('Image','PSF'):
        
        if units is None or units == '': # use native/attached values and units of image
            data = image.data.value
            units = image.data.unit
        else:
            aux = image.getBrightness(units)  # convert brightness to the desired units (at least try)
            data = aux.value
            units = str(aux.unit)

        # if field-of-view information is present in 'image', use it for extent; if 'extent' given as argument, ignore image.FOV
        if extent is None:
            fov, axunit = image.FOV.value, str(image.FOV.unit)
            rad = fov/2.
            extent = [-rad,rad,-rad,rad]
        
    elif cls == 'ndarray':
        data = image
        axunit = 'pixel'

    else:
        raise AttributeError("Don't know how to plot 'image'. Must be either instance of class 'Image', or 'PSF, or a 2d array.")

    # transpose here once for all future plotting in this function
    data = data.T
    
    # image normalization
    norm = matplotlib.colors.Normalize()  # possibly allow this to be an argument? (e.g. for absolute normalizations)
#    norm = matplotlib.colors.PowerNorm(1.)  # possibly allow this to be an argument? (e.g. for absolute normalizations)

    if cls == 'PSF':
        norm = matplotlib.colors.LogNorm()

    # reserve space for colorbar (even if later not used)
    divider = make_axes_locatable(ax)

    # plot image
    im = ax.imshow(data,origin='lower',extent=extent,interpolation=interpolation,cmap=cmap,norm=norm)

    # plot contours, if requested
    if contours is not None:
        if contours == 'log':
            aux = ma.masked_where(data <= 0, data)
            aux = N.log10(aux)
        elif contours == 'lin':
            aux = data

        ax.contour(aux,5,origin='lower',extent=extent,colors='w',linestyles='-')

    # set title, labels
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('offset ({:s})'.format(axunit))
    ax.set_ylabel('offset ({:s})'.format(axunit))

    # make colorbar; set invisible if no colorbar requested
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if colorbar == True:
        cb = p.colorbar(im,cax=cax,orientation='vertical')
        if cls == 'Image':
            cb.set_label(units)
    else:
        print("Setting cax.set_visible(False)")
        cax.set_visible(False)    


def multiplot(images,geometry=None,panelsize=4,direction='x',extent=None,\
              sharex=True,sharey=True,\
              colorbars=True,units='',titles='',contours=None,
              interpolation='bicubic',cmap=p.cm.viridis,figtitle='',\
              **kwargs):

    """Plot one or more images in a multi-panel figure.

    Flexible plotting of multi-panel figures.

    Parameters
    ----------
    images : instance or sequence
        Single instance of base class :class:`imageops.ImageFrame`
        object, or a sequence of them (list,tuple). If single
        instance, the multipanel figure will have shape (1,1).

        Note that instances of :class:`imageops.Image` will be
        normalized linearly, and those of :class:`imageops.Image`
        logarithmically.

    geometry : 2-tuple or None
        The (nrow,ncol) layout of the multipanel figure. If ``None``
        (default), the figure shape will be ``(1,len(images))``,
        i.e. all in one row. Otherwise `geometry` dictates the
        multiplot shape. ``np.prod(geometry)`` can be smaller or
        larger than ``len(images)``. If smaller, the displayed image
        sequence will be truncated. If larger, the image sequence will
        fill the first ``len(images)`` axes in the multiplot, leaving
        the other panels empty.

    panelsize : float
        Size of single panel in inches. Default is 4.

    direction : str
        If ``'x'`` (default), the figure panels are populated with
        images from the image sequence in the order
        row-first-then-column, i.e. left-to-right then
        top-to-bottom). If ``'y'``, the panels are populated
        column-first-then-row, i.e. top-to-bottom then left-to-right.

    extent : tuple or None
        Currently not used.

    sharex : bool
        If ``True`` (default), only the bottom row of panels will have
        x-axis labels and tick labels.

    sharey : bool
        If ``True`` (default), only the leftmost column of panels will
        have y-axis labels and tick labels.

    colorbars : bool or sequence of bools
        If ``True`` (default), add a colorbar to a panel. If single
        bool value, it determines for all panels whether colorbars
        will be added to each (i.e. ``True`` means add colorbars to
        all panels). If a sequence of bools, they determine the
        colorbar status for each panel separately. The number of
        booleans in the sequence `nb` need not be the same as the
        number of images in the `images` sequence `ni`; only
        ``max(nb,ni)`` elements will be used.

    units : str or sequence of strings
        String that describes to units of the colorbar quantity. If
        colorbar is ``True`` (for any panel), the corresponding
        element from the `units` sequence will be used at the
        colorbar label. The logic is the same as for `colorbars` (see
        there).

        Since the images in `images` carry units with them, the units
        in `units` will actually cause the plot to convert to the
        desired units.

    titles : str or sequence of strings
       Title(s) of each panel. Same logic as `colorbars` (see there).

    contours : str or sequence of strings or None
        If not None (default), add contours to panels. If string or
        seq of strings, they must be either ``'lin'`` for linearly
        spaced contours or ``'log'`` for logarithmically spaced
        contours. Five contour lines will be added. Same logic as
        `colorbars` (see there).

    interpolation : str
        Type of image interpolation to use on the panels. All values
        permissible my :func:`pylab.imshow()` are possible, including
        ``'none'``. Default is ``'bicubic'``.

    cmap : instance
        Colormap to be used for the images. Must be one of the
        colormaps in module :mod:`pylab.cm`. Default is
        ``p.cm.viridis``

    figtitle : str
        Super-title of the entire multi-panel figure. Default is
        ``''``.

    Examples
    --------

    Have a few :class:`imageops.ImageFrame` instances, here
    e.g. ``sky``, ``obs``, and ``psf``.

    .. code-block:: python

       # plots a 1x1 figure
       multiplot(img)

       # plots a 1x3 figure (1 row, 3 columns)
       multiplot((sky,obs,psf))

       # plots a 3x1 figure (3 rows, 1 columns)
       multiplot((sky,obs,psf),geometry=(3,1))

       # plots a 2x2 figure, the 1st row showing img & obs, 2nd row showing psf in the left panel
       multiplot((sky,obs,psf),geometry=(2,2))

       # plots a 2x2 figure, the 1st column showing img & obs, 2nd column showing psf in the upper panel
       multiplot((sky,obs,psf),geometry=(2,2),direction='y')

       # 1x3 figure, all panels have a colorbar (use same logic for ``units``, ``titles``, ``contours``)
       multiplot((sky,obs,psf),geometry=(1,3),colorbars=True)

       # 1x3 figure, 1st and last panels have a colorbar, middle panel doesn't (use same logic for ``units``, ``titles``, ``contours``)
       multiplot((sky,obs,psf),geometry=(1,3),colorbars=(True,False,True))

       # 1x3 figure, all panels have colorbar and (the same) units
       multiplot((sky,obs,psf),geometry=(1,3),colorbars=True,units=('Jy/arcsec^2','mJy/mas^2'))

       # 1x3 figure, all panels have colorbar; first two panels have colorbar units, albeit different ones
       multiplot((sky,obs,psf),geometry=(1,3),colorbars=True,units=('Jy/arcsec^2','mJy/mas^2'))

       # 1x3 figure; middle panel has 5 logarithmically scaled contours
       multiplot((sky,obs,psf),geometry=(1,3),contours=(None,'log'))

    """

    # IMAGES ARRAY
    images = arrayify(images,shape=geometry,fill=False,direction=direction)
    ny, nx = images.shape #geometry
    n = nx*ny

    # ARRAYS OF PANEL FEATURES
    colorbars = arrayify(colorbars,shape=images.shape,fill=True,direction=direction)
    units = arrayify(units,shape=images.shape,fill=True,direction=direction)
    titles = arrayify(titles,shape=images.shape,fill=True,direction=direction)
    contours = arrayify(contours,shape=images.shape,fill=True,direction=direction)

    # MAKE FIGURE
    # setup
    fontsize = 10
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['font.size'] =  fontsize
    p.rcParams['xtick.labelsize'] = fontsize-2
    p.rcParams['ytick.labelsize'] = fontsize-2
    p.rcParams['font.family'] = 'serif' # 'sans-serif'
#    # don't use Type 3 fonts (MNRAS requeirement)
#    p.rcParams['ps.useafm'] = True
#    p.rcParams['pdf.use14corefonts'] = True
#    p.rcParams['text.usetex'] = True

    figsize = (panelsize*nx,panelsize*ny)
    fig, axes = p.subplots(ny,nx,sharex=sharex,sharey=sharey,figsize=figsize)
    axes = N.atleast_2d(axes)
    if axes.T.shape == geometry:
        axes = axes.T
    
    # generate panels
    for iy in range(ny):
        for ix in range(nx):
            print("Plotting panel ({:d},{:d})".format(iy,ix))
            img = images[iy,ix]
            ax = axes[iy,ix]
            if img is not None:
                plotPanel(ax,img,units=units[iy,ix],colorbar=colorbars[iy,ix],title=titles[iy,ix],cmap=cmap,contours=contours[iy,ix],interpolation=interpolation)
            else:
                ax.set_visible(False)

    # only lower panels get x-label
    if sharex is True:
        for ax in axes[:-1,:].flatten():
            ax.set_xlabel('')
            
    # only left-most panels get y-label
    if sharey is True:
        for ax in axes[:,1:].flatten():
            ax.set_ylabel('')

    # figure super title
    fig.suptitle(figtitle)


    # TODO: fix this; only check for the adjustment keys in kwargs (i.e. left,right, etc)
    # fig.subplots_adjust() arguments; can be modified via kwargs
    adjustkwargs = {'left':0.15,'right':0.88,'top':0.97,'bottom':0.06,'hspace':0.15,'wspace':0.35}
    for k,v in kwargs.items():
        adjustkwargs[k] = v
        
    fig.subplots_adjust(**adjustkwargs)
    
    return fig, axes
