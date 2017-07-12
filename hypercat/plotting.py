__version__ = '20170208'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

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
#import astropy

# own
from utils import *

def plotPanel(ax,image,units='',extent=None,colorbar=False,title='',cmap=p.cm.viridis,contours=None):

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
        raise AttributeError, "Don't know how to plot 'image'. Must be either instance of class 'Image', or 'PSF, or a 2d array."

    # transpose here once for all future plotting in this function
    data = data.T
    
    # image normalization
    norm = matplotlib.colors.Normalize()  # possibly allow this to be an argument? (e.g. for absolute normalizations)

    if cls == 'PSF':
        norm = matplotlib.colors.LogNorm()

    # reserve space for colorbar (even if later not used)
    divider = make_axes_locatable(ax)

    # plot image
    im = ax.imshow(data,origin='lower',extent=extent,interpolation='none',cmap=cmap,norm=norm)

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
    ax.set_xlabel('offset (%s)' % axunit)
    ax.set_ylabel('offset (%s)' % axunit)

    # make colorbar; set invisible if no colorbar requested
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if colorbar == True:
        cb = p.colorbar(im,cax=cax,orientation='vertical')
        if cls == 'Image':
            cb.set_label(units)
    else:
        cax.set_visible(False)    

            
def multiplot(images,geometry=None,panelsize=4,direction='x',extent=None,\
              sharex=True,sharey=True,\
              colorbars=True,\
              units='',\
              titles='',\
              contours=None,
              interpolation='none',cmap=p.cm.viridis,figtitle='',\
              **kwargs):

    # IMAGES ARRAY
    images = arrayify(images,shape=geometry,direction=direction)
    
    ny, nx = images.shape #geometry
    n = nx*ny

    # ARRAYS OF PANEL FEATURES
    colorbars = arrayify(colorbars,shape=geometry,direction=direction)
    units = arrayify(units,shape=geometry,direction=direction)
    titles = arrayify(titles,shape=geometry,direction=direction)
    contours = arrayify(contours,shape=geometry,direction=direction)
    
    # MAKE FIGURE
    # setup
    fontsize = 10
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['text.fontsize'] =  fontsize
    p.rcParams['xtick.labelsize'] = fontsize-2
    p.rcParams['ytick.labelsize'] = fontsize-2
    p.rcParams['font.family'] = 'serif' # 'sans-serif'
#    # don't use Type 3 fonts (MNRAS requeirement)
#    p.rcParams['ps.useafm'] = True
#    p.rcParams['pdf.use14corefonts'] = True
#    p.rcParams['text.usetex'] = True

    figsize = (panelsize*nx,panelsize*ny)
    fig, axes = p.subplots(ny,nx,sharex=True,sharey=True,figsize=figsize)
    axes = N.atleast_2d(axes)
    if axes.T.shape == geometry:
        axes = axes.T
    
    # generate panels
    for iy in xrange(ny):
        for ix in xrange(nx):
            print "Plotting panel (%d,%d)" % (iy,ix)
            img = images[iy,ix]
            ax = axes[iy,ix]
            if img is not None:
                plotPanel(ax,img,units=units[iy,ix],colorbar=colorbars[iy,ix],title=titles[iy,ix],cmap=cmap,contours=contours[iy,ix])
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

    # fig.subplots_adjust() arguments; can be modified via kwargs
    adjustkwargs = {'left':0.15,'right':0.88,'top':0.97,'bottom':0.06,'hspace':0.15,'wspace':0.35}
    for k,v in kwargs.items():
        adjustkwargs[k] = v
        
    fig.subplots_adjust(**adjustkwargs)
    
    return fig, axes
