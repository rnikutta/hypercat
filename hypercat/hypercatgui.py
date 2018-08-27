"""GUI interface to (some) Hypercat functionality."""

__version__ = '20180826'
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

# std lib
import time
from copy import copy
import string
from random import *
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import shutil
import subprocess

# 3rd party
import numpy as np
import pylab as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter as Tk
from tkinter import ttk, filedialog, messagebox

import astropy
from astropy.io import fits
from astropy.samp import SAMPIntegratedClient

# Hypercat
import hypercat
import imageops


class App():

    def __init__(self):

        self.root = Tk.Tk()
        
        self.bgcolor = self.root.cget('bg')
        self.root.configure(background=self.bgcolor)
        self.root.title("Hypercat")
        self._geometry()
        
        self.hdf5file = None
#        self.hdf5file = '/home/robert/data/hypercat/hypercat_20180417.hdf5'
        self.ds9process = None

        # Create tabs
        self.nb = ttk.Notebook(self.root)
        self.add_elem('tabM',ttk.Frame,self.nb,state='normal',add=True,title='Model')
        self.add_elem('tabS',ttk.Frame,self.nb,state='disabled',add=True,title='Single-dish')
        self.add_elem('tabI',ttk.Frame,self.nb,state='disabled',add=True,title='Interferometry')
        self.add_elem('tabIFU',ttk.Frame,self.nb,state='disabled',add=True,title='IFU')
        self.add_elem('tabMorpho',ttk.Frame,self.nb,state='disabled',add=True,title='Morphology')

        ### tabM
        # Select and load HDF5 file
        self.l1 = Tk.Label(self.tabM,text="HDF5 file")
        self.l1.grid(row=0,column=0,sticky='W')
        self.v1 = Tk.StringVar(self.tabM,value='Pick HDF5 file') #, self.root, value=self.hdf5file)
#        self.v1.trace('w',self.setvar)
        self.e1 = Tk.Entry(self.tabM,state='readonly',width=50,textvariable=self.v1)
        self.e1.grid(row=0,column=1,columnspan=5,sticky='W')
        self.b1 = Tk.Button(self.tabM,text="Pick file",command=self.pick_and_load)
        self.b1.grid(row=0,column=1+5,sticky='W')

        # Select colormap and reload image
        l2 = Tk.Label(self.tabM,text="Colormap")
        l2.grid(row=1,column=0,sticky='W')
        self.varcmap = Tk.StringVar()
        cmaps = ('gray', 'viridis', 'jet', 'inferno', 'cubehelix', 'cividis', 'afmhot', 'bwr','coolwarm','gnuplot2','rainbow')
        self.cmapMenu = Tk.OptionMenu(self.tabM, self.varcmap, *cmaps, command=self.update_view)
        self.varcmap.set('inferno')
        self.cmapMenu.grid(row=1, column=1, columnspan=3, sticky='ew')

        # Checkbox to invert colormap
        self.varInvert = Tk.IntVar()
        self.cmapInvert = Tk.Checkbutton(self.tabM, text="invert", variable=self.varInvert, onvalue=1, padx=10, command=self.update_view)
        self.cmapInvert.deselect()
        self.cmapInvert.grid(row=1,column=4,columnspan=1,sticky='W')

        # Select linear or log colormap normalization
        self.varNorm = Tk.StringVar()
        self.cmapNorm1 = Tk.Radiobutton(self.tabM, text="linear", variable=self.varNorm, value='Normalize',pady=0,command=self.update_view)
        self.cmapNorm1.grid(row=1,column=4,columnspan=1,sticky='e')
        self.cmapNorm1.select()
        self.cmapNorm2 = Tk.Radiobutton(self.tabM, text="log", variable=self.varNorm, value='LogNorm',pady=0,command=self.update_view)
        self.cmapNorm2.grid(row=1,column=5,columnspan=1,sticky='w')
        self.cmapNorm2.deselect()
        
        # ask for HDF5 file and load cube
        if not self.hdf5file:
            self.pick_and_load(text='Select Hypercat HDF5 file')

        # (linear) sliders for model parameters
        row = 4
#        formats = ('%d','%d','%.1f','%.1f','%.1f','%d') # sig, i, Y, N, q, tv
        formats = ('%d','%d','%d','%.1f','%.1f','%d') # sig, i, Y, N, q, tv
        labels = (' (deg)',' (deg)','','','','')
#        resolutions = (1,1,0.1,0.2,0.1,5)
        resolutions = (1,1,1,0.2,0.1,5)
        inits = (54,75,18,7,0,80)
        for j,par in enumerate(self.cube.paramnames[:6]):
            theta = self.cube.theta[j]
            MIN, MAX = theta[0], theta[-1]
            objname = 'scale_'+par
            setattr(self,objname,LinSlider(self.tabM,MIN,MAX,resolutions[j],label=par+labels[j],fmt=formats[j]))
            obj = getattr(self,objname)
            obj.grid(row=row,column=2+20,columnspan=2,sticky='W')
            obj.set(inits[j])
            row += 1
            
        # log slider for wavelength
        self.scale_wave = LogSlider(self.tabM,from_=1.2,to=870.)
        self.scale_wave.grid(row=row,column=2+20,columnspan=2,sticky='W')
        self.scale_wave.set(np.log10(10.2))
                    
        # lin slider for PA
        self.scalePA = LinSlider(self.tabM,90,-90,1,label='PA (deg E of N)',fmt='%d')
        self.scalePA.grid(row=row+1,column=2+20,columnspan=2,sticky='W')
        self.scalePA.set(42)

        # button to update image
        button_update = Tk.Button(self.tabM, text="Update image", command=self.update_image)
        button_update.grid(row=row+2,column=2+20,columnspan=1,sticky='W')
        
        button_ds9 = Tk.Button(self.tabM, text="View in DS9", command=self.send2ds9)
        button_ds9.grid(row=row+2,column=3+20,columnspan=1,sticky='E')

        # load first image
        self.update_image()

        # make all
        self.nb.pack(expand=1, fill="both")  # Pack to make visible
#        self.root.update()
        

    def _geometry(self):
        w = 700 # width for the Tk root
        h = 600 # height for the Tk root

        # get screen width and height
        ws = self.root.winfo_screenwidth() # width of the screen
        hs = self.root.winfo_screenheight() # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        # set the dimensions of the screen 
        # and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (w,h,x,y))
        

    def update_view(self,*args):

        invert = self.varInvert.get()
        cmap = self.varcmap.get()
        if invert == 1:
            cmap = cmap + '_r'
        cmap = getattr(plt.cm,cmap)
        norm = getattr(plt.mpl.colors,self.varNorm.get())
        
        fig = Figure(figsize=(4.8,4.8), facecolor=self.bgcolor)
        fig.add_subplot(111).imshow(self.img.T,origin='lower',interpolation='bicubic',cmap=cmap,norm=norm())
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.tabM)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().grid(row=2,column=0,columnspan=20,sticky='W',rowspan=20)
        self.canvas.draw()

        self.toolbar_frame = ttk.Frame(self.tabM)
        self.toolbar_frame.grid(row=22,column=0,columnspan=30,sticky='W')
        self.toolbar = NavigationToolbar2Tk(self.canvas,self.toolbar_frame)

        
    def update_image(self):
        sig = self.scale_sig.get()
        i = self.scale_i.get()
        Y = self.scale_Y.get()
        N = self.scale_N.get()
        q = self.scale_q.get()
        tv = self.scale_tv.get()
        wave = 10**self.scale_wave.get()
        vec = tuple([sig,i,Y,N,q,tv,wave])
        print("vec = ", vec)
        self.img = self.cube(vec)

        pa = self.scalePA.get()
        if pa != 0:
            self.img = imageops.rotateImage(self.img,'%d deg' % pa)
        
        self.update_view()


    def pick_and_load(self,text=''):
        self.pick_file(text=text)
        self.load_cube()
                    
        
    def pick_file(self,target='hdf5file',text=''):
        filename = filedialog.askopenfilename(title=text)
        if target is not None:
            setattr(self,target,filename)
            self.v1.set(self.hdf5file)

        
    def load_cube(self):
        self.cube = hypercat.ModelCube(self.hdf5file,hypercube='imgdata',subcube_selection='onthefly')


    def launch_samphub(self):
        from astropy.samp import SAMPHubServer
        self.hub = SAMPHubServer(web_profile=False)
        self.hub.start()
        

    def add_elem(self,name,what,target,state='normal',add=False,title='',**kwargs):
        obj = what(target,**kwargs)
        if add is True:
            target.add(obj,text=title,state=state)
            
        setattr(self,name,obj)
        
        
    def make_hdu(self):
        hdu = fits.PrimaryHDU(data=self.img.T)
        return hdu
    

    def send2ds9(self):

        self.ds9 = shutil.which("ds9")
        if self.ds9 is None:
            messagebox.showerror("Error", "No DS9 program found in $PATH.")
            return
            
        self.update_image()
        
        self.client = SAMPIntegratedClient()

        try:
            self.client.connect()
        except astropy.samp.errors.SAMPHubError:
            self.launch_samphub()
            self.client.connect()

        if self.ds9process is None or self.ds9process.poll() == 0:
            self.ds9process = subprocess.Popen(self.ds9) #"/home/robert/src/sao_ds9_7.6/ds9")
            time.sleep(3) # allow DS9 to launch fully

        # creat temp fits file, notify SAMP hub, clean up temp file
        f = NamedTemporaryFile(mode='wb',delete=False)
#P        print("os.path.isfile(f.name)", os.path.isfile(f.name))
        hdu = self.make_hdu()
        hdu.writeto(f)
        
        params = {}
        params["url"] = 'file://' + f.name
        params["name"] = "Hypercat image"        
        message = {}
        message["samp.mtype"] = "image.load.fits"
        message["samp.params"] = params

        self.client.notify_all(message)
        time.sleep(1)
        f.close()
        os.unlink(f.name)
#P        print("os.path.isfile(f.name)", os.path.isfile(f.name))



#    def on_key_press(self,event):
#        print("you pressed {}".format(event.key))
#        key_press_handler(event, self.canvas, toolbar)


    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate

                        

class LinSlider(Tk.Scale):
    def __init__(self,target,from_,to,delta,label='',fmt='%.1f'):

        self.fmt = fmt
        
        Tk.Scale.__init__(self,target,from_=from_,to=to,length=180,\
                          digits=5,resolution=delta,orient='horizontal',showvalue=0,label=label,\
                          command=self._update_label)
        val = 0.5*(from_+to)
        self.set(val)
        self._update_label(val)

    def _update_label(self,val):
        print("val = ", val)
        update_label(self,val,fmt=self.fmt,log=False)


class LogSlider(Tk.Scale):
    def __init__(self,target,from_,to,label='wave (micron)'):
        
        logfrom = np.log10(from_) 
        logto = np.log10(to)

        epsmin = logfrom*1e-6
        epsmax = logto*1e-6
        
        Tk.Scale.__init__(self,target,from_=logfrom+epsmin,to=logto-epsmax,length=180,\
                          digits=5,resolution=0.000001,orient='horizontal',showvalue=0,label=label,\
                          command=self._update_label)
        self.set(0.5*(logfrom+logto))

    def _update_label(self,val):
        print("val = ", val)
        update_label(self,val,log=True)

        
def update_label(obj,val,fmt='%.1f',log=False,limits=None):

    val = float(val)
    
    if log is True:
        val = 10**val

    if limits is not None:
        MIN, MAX = limits
        val = max(MIN,val)
        val = min(MAX,val)
        
    label = obj.cget('label').split('=')[0].rstrip()
    label = label + " = %s" % fmt % val
    obj.configure(label=label)
    print(label)

    

if __name__ == '__main__':
    app = App()
    app.root.mainloop()
    

#Tk.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
