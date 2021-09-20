#!/usr/bin/env python

"""GUI interface to (some of) Hypercat functionality."""

__version__ = '20210917'
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

# std lib
import time
import string
import os
from tempfile import NamedTemporaryFile
import shutil
import subprocess
import configparser

# 3rd party
import numpy as np
import pylab as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as Tk
from tkinter import ttk, filedialog, messagebox, font

import astropy
from astropy.io import fits
from astropy.samp import SAMPIntegratedClient

# Hypercat
from hypercat.hypercat import ModelCube
from hypercat import imageops
from hypercat import ioops
from hypercat.loggers import *

CONFIGFILE = 'hypercatgui.conf'

def read_or_create_config():
    config = configparser.ConfigParser()
    config.optionxform = str  # deactivates lower-casing of keys and values
    
    defaults = {'hdf5file': '',
                'colormap': 'inferno',
                'scale': 'linear',
                'invert': '0',
                'sig': '43',
                'i': '75',
                'Y': '18',
                'N': '7',
                'q': '0.0',
                'tv': '70',
                'wave': '10.0',
                'PA': '42'}
    
    writeconfigfile = False

    print("list(config.keys()): ", list(config.keys()))
    print("config['DEFAULT']: ", config['DEFAULT'])
    print("list(config['DEFAULT'].keys()): ", list(config['DEFAULT'].keys()))
    
    if config.read(CONFIGFILE) == []:
        print("No config file found. Creating default config file %s" % CONFIGFILE)
        config['DEFAULT'] = defaults
        config['USER'] = defaults
        writeconfigfile = True

    for section in ('DEFAULT','USER'):
        print("section: ", section)
        if section not in config or list(config[section].keys()) == []:
            print("No %s section found in config file %s, or it is empty. Creating/populating." % (section,CONFIGFILE))
            config[section] = defaults
            writeconfigfile = True

    if writeconfigfile is True:
        with open(CONFIGFILE, 'w') as configfile:
            config.write(configfile)

    # add TEMP section to config, but only for runtime (i.e. don't save to config file)
    config['TEMP'] = config['USER']

    return config


def logmsgbox(loglevel='info',title='',msg=''):

    """Log an info/warn/error message, while also showing it in a message box.

    Parameters:
    -----------

    loglevel : str
        One of 'info' (default), 'warn', 'error', i.e. log levels from
        the logging module. The levels are mapped to the tkinter
        messagebox methods 'showinfo', 'showwarning', 'showerror'.

    title : str
        Title of the message box. Defaults to ''.

    msg : str
        The message to be logged, and displayed in the message box.

    Examples:
    ---------

    logmsgbox('info','Information','You are doing good.')
    logmsgbox('warn','Warning','This is a warning message')
    logmsgbox('error','Error','Oh-oh!')

    """
    
    # mapping logging levels to tkinter messagebox methods
    loglevels = {'info' : 'showinfo', 'warn' : 'showwarning', 'error' : 'showerror'}

    # get the methods to log and to show a msg box
    logmethod = getattr(logging,loglevel)
    msgboxmethod = getattr(messagebox,loglevels[loglevel])

    # log the msg and pop up a msg box
    logmethod(msg)
    msgboxmethod(title,msg)
    
        
class App():


    def __init__(self,config):

        self.config = config
        print("self.config: ", self.config)

        self.root = Tk.Tk()
        
#        default_font = tkFont.nametofont("TkDefaultFont")
#        default_font = font.nametofont("TkFixedFont")
##        default_font.configure(size=48)
##        self.default_font = tkFont.Font(family="Verdana", size=10)
#G        self.root.default_font = font.Font(family="Times", size=16)
#        self.root.default_font = font.Font("TkFixedFont", size=12)
##tkFont.Font(family="{Times}",size=20)

        # fonts for all widgets
        self.root.option_add("*Font", "helvetica 11")

        # font to use for label widgets
        self.root.option_add("*Label.Font", "helvetica 11")

        
        self.bgcolor = self.root.cget('bg')
        self.root.configure(background=self.bgcolor)
        self.root.title("Hypercat")
        self._geometry()

        self.ds9process = None

        # Create tabs
        self.nb = ttk.Notebook(self.root)
        self.add_elem('tabM',ttk.Frame,self.nb,state='normal',add=True,title='Model')
        self.add_elem('tabS',ttk.Frame,self.nb,state='normal',add=True,title='Single-dish')
        self.add_elem('tabI',ttk.Frame,self.nb,state='disabled',add=True,title='Interferometry')
        self.add_elem('tabIFU',ttk.Frame,self.nb,state='disabled',add=True,title='IFU')
        self.add_elem('tabMorpho',ttk.Frame,self.nb,state='normal',add=True,title='Morphology')

        ### tabM
        # Select and load HDF5 file
        self.l1 = Tk.Label(self.tabM,text="HDF5 file")
        self.l1.grid(row=0,column=0,sticky='W')
        self.v1 = Tk.StringVar(self.tabM,value='Pick HDF5 file') #, self.root, value=self.hdf5file)
        self.e1 = Tk.Entry(self.tabM,state='readonly',width=50,textvariable=self.v1)
        self.e1.grid(row=0,column=1,columnspan=5,sticky='W')
        self.b1 = Tk.Button(self.tabM,text="Pick file",command=self.pick_and_load)
        self.b1.grid(row=0,column=1+5,columnspan=2,sticky='W')

        # Select colormap and reload image
        l2 = Tk.Label(self.tabM,text="Colormap")
        l2.grid(row=1,column=0,sticky='W')
        self.varcmap = Tk.StringVar()
        cmaps = ('gray', 'viridis', 'jet', 'inferno', 'cubehelix', 'cividis', 'afmhot', 'bwr','coolwarm','gnuplot2','rainbow')
        self.cmapMenu = Tk.OptionMenu(self.tabM, self.varcmap, *cmaps, command=self.update_image)
        try:
            self.varcmap.set(config['TEMP']['colormap'])
        except:
            self.varcmap.set('inferno')
            
        self.cmapMenu.grid(row=1, column=1, columnspan=3, sticky='ew')

        # Checkbox to invert colormap
        self.varInvert = Tk.IntVar()
        self.cmapInvert = Tk.Checkbutton(self.tabM, text="invert", variable=self.varInvert, onvalue=1, padx=10, command=self.update_image)
        if config['TEMP']['invert'] == '1':
            self.cmapInvert.select()
        else:
            self.cmapInvert.deselect()
        self.cmapInvert.grid(row=1,column=4,columnspan=1,sticky='W')

        # Select linear or log colormap normalization
        self.varNorm = Tk.StringVar()
        self.cmapNorm1 = Tk.Radiobutton(self.tabM, text="linear", variable=self.varNorm, value='linear',pady=0,command=self.update_image)
        self.cmapNorm1.grid(row=1,column=4,columnspan=1,sticky='e')
        if config['TEMP']['scale'] == 'linear':
            self.cmapNorm1.select()
        else:
            self.cmapNorm1.deselect()
            
        self.cmapNorm2 = Tk.Radiobutton(self.tabM, text="log", variable=self.varNorm, value='log',pady=0,command=self.update_image)
        self.cmapNorm2.grid(row=1,column=5,columnspan=1,sticky='w')
        if config['TEMP']['scale'] == 'log':
            self.cmapNorm2.select()
        else:
            self.cmapNorm2.deselect()
        
        self.hdf5file = config['TEMP']['hdf5file']
        if not os.path.isfile(self.hdf5file):
            self.pick_and_load(text='Select Hypercat HDF5 file')
            config['TEMP']['hdf5file'] = self.hdf5file
        else:
            self.load_cube()

        self.v1.set(self.hdf5file)

        # (linear) sliders for model parameters
        row = 4
#        formats = ('%d','%d','%.1f','%.1f','%.1f','%d') # sig, i, Y, N, q, tv
#        formats = ('%d','%d','%d','%.1f','%.1f','%d') # sig, i, Y, N, q, tv
#        self.labels = ('deg','deg','','','','','mu','deg')
#        resolutions = (1,1,0.1,0.2,0.1,5)
#        resolutions = (1,1,1,0.2,0.1,5)

        self.parnames = ('sig','i','Y','N','q','tv','wave','PA')
        self.increments = (1,1,1,0.1,0.1,1,0.1,1)
        self.ranges = [(self.cube.theta[j][0],self.cube.theta[j][-1]) for j in range(7)]
        self.ranges.append((-360,360))
        self.units = ('deg','deg','R_d','clouds','','','mu','deg')
        print("self.ranges: ", self.ranges)

        ParamsLabel = Tk.Label(self.tabM,text="Parameters [min - max] (units)")
        ParamsLabel.grid(row=row,column=8,columnspan=2,sticky='W')
        row += 1

        vcmd = (self.root.register(self.ValidateIfNum), '%s', '%S', '%d', '%i', '%P', '%v', '%V', '%W')
        for j in range(8):
            self.make_spinbox(j,self.increments[j],row,8,vcmd)
            row += 1

        # set param values
        self.set_values()



        # button to update image
        button_update = Tk.Button(self.tabM, text="Update image", width=12, command=self.update_image)
        self.root.bind('<Return>', self.update_image) # experimental
        button_update.grid(row=row+2,column=8,columnspan=2,sticky='W')
        
        button_ds9 = Tk.Button(self.tabM, text="View in DS9 ", width=12, command=self.send2ds9)
        button_ds9.grid(row=row+4,column=8,columnspan=2,sticky='W')

        button_fits = Tk.Button(self.tabM, text="Save as FITS", width=12, command=self.save2fits)
        button_fits.grid(row=row+5,column=8,columnspan=2,sticky='W')

        # load first image
        self.update_image('foo')

        # make all
        self.nb.pack(expand=1, fill="both")  # Pack to make visible


    def initialize(self):
        pass
        
    def ValidateIfNum(self, s, S, d, i, P, v, V, W):
        print("s: ", s)
        print("S: ", S)
        print("d: ", d)
        print("i: ", i)
        print("P: ", P)
        print("v: ", v)
        print("V: ", V)
        print("W: ", W)

        print(self.root.nametowidget(W))


        from_ = self.root.nametowidget(W).config('from')[4]
        to_ = self.root.nametowidget(W).config('to')[4]
        print("from_, to_ = ", from_, to_)

        isvalid = False

        if P in ('','-','+','.') or self.isNumber(P):
            isvalid = True

        return isvalid

                
    def isNumber(self,arg):
        try:
            float(arg)
            return True
        except:
            return False


    def onValidate(self, d, i, P, s, S, v, V, W):
#        self.text.delete("1.0", "end")
#        self.text.insert("end","OnValidate:\n")
#        self.text.insert("end","d='%s'\n" % d)
#        self.text.insert("end","i='%s'\n" % i)
        self.text.insert("end","P='%s'\n" % P)
        self.text.insert("end","s='%s'\n" % s)
        self.text.insert("end","S='%s'\n" % S)
#        self.text.insert("end","v='%s'\n" % v)
#        self.text.insert("end","V='%s'\n" % V)
#        self.text.insert("end","W='%s'\n" % W)

#        self.delete("1.0", "end")
#        self.insert("end","OnValidate:\n")
#        self.insert("end","d='%s'\n" % d)
#        self.insert("end","i='%s'\n" % i)
#        self.insert("end","P='%s'\n" % P)
#        self.insert("end","s='%s'\n" % s)
#        self.insert("end","S='%s'\n" % S)
#        self.insert("end","v='%s'\n" % v)
#        self.insert("end","V='%s'\n" % V)
#        self.insert("end","W='%s'\n" % W)

        print("self:", self)
        print("self.from_:", self.from_)
        return True


    def _geometry(self):
        w = 800 # width for the Tk root
        h = 700 # height for the Tk root

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
        
        norms = {'linear':'Normalize', 'log':'LogNorm'}
        norm = getattr(plt.mpl.colors,norms[self.varNorm.get()])
        
        fig = Figure(figsize=(5.5,5.5), facecolor=self.bgcolor)
        ax = fig.add_subplot(111)
        ax.imshow(self.img.T,origin='lower',interpolation='bicubic',cmap=cmap,norm=norm())

        figS = Figure(figsize=(4,4), facecolor=self.bgcolor)
        axS = figS.add_subplot(111)
        axS.imshow(self.img.T,origin='lower',interpolation='bicubic',cmap=cmap,norm=norm())

        figMorpho = Figure(figsize=(3,3), facecolor=self.bgcolor)
        axMorpho = figMorpho.add_subplot(111)
        axMorpho.imshow(self.img.T,origin='lower',interpolation='bicubic',cmap=cmap,norm=norm())

        fig.tight_layout()
        figS.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.tabM)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().grid(row=2,column=0,columnspan=20,sticky='W',rowspan=20)
        self.canvas.draw()

        self.toolbar_frame = ttk.Frame(self.tabM)
        self.toolbar_frame.grid(row=22,column=0,columnspan=30,sticky='W')
        self.toolbar = NavigationToolbar2Tk(self.canvas,self.toolbar_frame)

        # Duplicate canvas on tabPSF
        self.canvasS = FigureCanvasTkAgg(figS, master=self.tabS)  # A tk.DrawingArea.
        self.canvasS.get_tk_widget().grid(row=2,column=0,columnspan=20,sticky='W',rowspan=20)
        self.canvasS.draw()
        
        # Duplicate canvas on tabMorpho
        self.canvasMorpho = FigureCanvasTkAgg(figMorpho, master=self.tabMorpho)  # A tk.DrawingArea.
        self.canvasMorpho.get_tk_widget().grid(row=2,column=0,columnspan=20,sticky='W',rowspan=20)
        self.canvasMorpho.draw()
#        self.canvasMorpho.resize(0.5)

#G        self.toolbar_frameMorpho = ttk.Frame(self.tabMorpho)
#G        self.toolbar_frameMorpho.grid(row=22,column=0,columnspan=30,sticky='W')
#G        self.toolbarMorpho = NavigationToolbar2Tk(self.canvasMorpho,self.toolbar_frameMorpho)


    def get_vector(self,source='config'):
        
        if source == 'config':
            vec = tuple([config['TEMP'][par] for par in self.parnames])

        elif source == 'vars':
            vec = tuple([getattr(self,par + 'SB').Var.get() for par in self.parnames])
            
        return vec
    

    def update_image(self,event=None):

        vec = self.get_vector(source='vars')

        try:
            self.img = self.cube(vec[:-1]) # omit the last element, PA
        except:
            logmsgbox('warn','Warning','Invalid model parameter(s) encountered. Reverting to last valid parameter set.')
            self.set_values() # setting widgets back to last valid values, using config['TEMP']
        else:
            pa = self.PASB.Var.get()
            if pa != 0:
                self.img = imageops.rotateImage(self.img,'%d deg' % pa)
        finally:
#            print("Interpolation OK. Saving new values to config TEMP section")
            self.update_temp_config()
            self.update_view()


    def pick_and_load(self,text=''):
        self.pick_file(text=text)
        self.load_cube()
                    
        
    def pick_file(self,target='hdf5file',text=''):
        self.hdf5file = filedialog.askopenfilename(title=text)
        if target is not None:
            setattr(self,target,self.hdf5file)
            self.v1.set(self.hdf5file)

        
    def load_cube(self):
#        self.cube = hypercat.ModelCube(self.hdf5file,hypercube='imgdata',subcube_selection='onthefly')
        self.cube = ModelCube(self.hdf5file,hypercube='imgdata',subcube_selection='onthefly')


    def launch_samphub(self):
        from astropy.samp import SAMPHubServer
        self.hub = SAMPHubServer(web_profile=False)
        self.hub.start()
        

    def add_elem(self,name,what,target,state='normal',add=False,title='',**kwargs):
        obj = what(target,**kwargs)
        if add is True:
            target.add(obj,text=title,state=state)
            
        setattr(self,name,obj)


    def make_spinbox(self,j,increment,row,col,vcmd):
        par = self.parnames[j]
        print("par: ", par)

        from_, to_ = self.ranges[j][0], self.ranges[j][-1]
        widgetname = par + 'SB'
        
        unit = self.units[j]
        if unit != '':
            unit = "(%s)" % unit

        if par == 'PA':
            labeltext = "PA [any] %s" % unit
        else:
            labeltext = "%s [%g - %g] %s" % (par,from_,to_,unit)

        section = 'USER' if 'USER' in self.config else 'DEFAULT'
        init = float(self.config[section][par])
        spinbox = BetterSpinbox(self.tabM,par,from_,to_,init,increment,labeltext,vcmd)
        spinbox.grid(row=row,column=col,sticky='W')
        spinbox.Label.grid(row=row,column=col+1,sticky='W')
        setattr(self,widgetname,spinbox)
        
        
    def save2fits(self):

        fname = filedialog.asksaveasfilename(defaultextension=".fits")
        
        if fname is None:
            messagebox.showerror("Error", "No filename provided.")
            return
        
        if not fname.endswith('.fits'):
            fname += '.fits'

        I = imageops.Image(self.img)
        ioops.save2fits(I,fname)
        print("Saved to file %s" % fname)
        

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


    def _quit(self):
        self.save_config()
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate


    def update_temp_config(self):
        vec = self.get_vector(source='vars')
        for j,par in enumerate(self.parnames):
            config['TEMP'][par] = str(vec[j])

        config['TEMP']['colormap'] = self.varcmap.get()
        config['TEMP']['scale'] = self.varNorm.get()
        config['TEMP']['invert'] = str(self.varInvert.get())

    def set_values(self):
        vec = self.get_vector(source='config')
        for j,par in enumerate(self.parnames):
            getattr(self,par + 'SB').Var.set(float(vec[j]))
        
    def save_config(self):
        self.config['USER'] = self.config['TEMP']
        self.config.pop('TEMP') # remove TEMP section before saving config file
        with open(CONFIGFILE, 'w') as configfile:
                self.config.write(configfile)

        
class BetterSpinbox(Tk.Spinbox):

    def __init__(self,parent,name,from_,to_,init,increment,labeltext='',vcmd=None):
        self.Var = Tk.DoubleVar()
        Tk.Spinbox.__init__(self,parent,from_=from_,to=to_,increment=increment,width=5,textvariable=self.Var,validate="key",validatecommand=vcmd)

#        myfont = font.Font(family='Helvetica', size=12) # , weight='bold'
        self.Label = Tk.Label(parent,text=labeltext) # ,font=myfont
#        self.Label = Tk.Label(parent,text=labeltext,font = ('Times',12))


def main():
    global config, app
    config = read_or_create_config()
    app = App(config)
    app.root.protocol("WM_DELETE_WINDOW", app._quit)
    app.root.mainloop()
    

if __name__ == '__main__':
    main()
    
#    app.root.protocol("WM_DELETE_WINDOW", app._quit)
##    app.bind("<Destroy>", _destroy)


#Tk.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
