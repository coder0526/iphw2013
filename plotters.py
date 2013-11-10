# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import numpy as np

class CrazyPlotter(object):
    def __init__(self):
        self.data = 'Grid plot class'
                
    def _hide_ticks(self, fig, x=True, y=True):
        for i, ax in enumerate(fig.axes):
            #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            #ax.set_title('good%d' % i)
            if x: ax.xaxis.set_ticks([])
            if y: ax.yaxis.set_ticks([])
            
    def _hist(self, axis, img):
        color = ('b','g','r')
        if img.ndim == 2:
            #histr = cv2.calcHist([img],[0],None,[256],[0,256])
            #axis.plot(histr, color='gray')
            axis.hist(img[...,0].flatten(), 256, range=(0, 256), fc='gray', alpha=.5)
        elif img.ndim == 3:
            for i in range(0, img.ndim):
                #histr = cv2.calcHist([img],[i],None,[256],[0,256])
                #axis.plot(histr, color=color[i])
                axis.hist(img[...,i].flatten(), 128, fc=color[i], alpha=.5)
                
            
    def _plot_histo(self, fig, index, img, title='untitled'):
        grid = GridSpec(4, 3)
        ax1 = fig.add_subplot(grid[:3, index])
        ax1.set_title(title)
        ax1.imshow(img, cmap="Greys_r")
        ax2 = fig.add_subplot(grid[3:, index])
        ax2.set_title(title+'-histogram')
        self._hist(ax2, img)
        ax2.set_xlim(0, 256)

    def histo_equal(self, **kwargs): #image processing
        title = kwargs.get('title', 'untitled')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(hspace=.14, wspace=.03)
        fig.suptitle( title, fontsize=25, y=.96)
        self._plot_histo(fig, 0, img, 'original')
        gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self._plot_histo(fig, 1, gimg, 'grayed')
        egimg = cv2.equalizeHist(gimg)
        self._plot_histo(fig, 2, egimg, 'equalized')
        self._hide_ticks(fig, x=False)
        return fig
        
    def _plot(self, fig, index, img, title='untitled'):
        ax1 = fig.add_subplot(2,2,index)
        ax1.set_title(title)
        if img is not None: ax1.imshow(img, cmap="Greys_r")
    def _plot2(self, fig, index, img, title='untitled'):
        ax1 = fig.add_subplot(1,3,index)
        ax1.set_title(title)
        if img is not None: ax1.imshow(img, cmap="Greys_r")

    def blurring(self, **kwargs): #image processing
        title = kwargs.get('title', 'Blurring')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=.1, wspace=.03)
        fig.suptitle( title, fontsize=25, y=.96)
        #core
        self._plot(fig, 1, img, 'original')
        bimg = cv2.boxFilter(img, 0, (15, 15))
        self._plot(fig, 2, bimg, 'Averaging(Mean) Filtering')
        mimg = cv2.medianBlur(img, 15) 
        self._plot(fig, 3, mimg, 'Median Filtering')
        gimg = cv2.GaussianBlur(img, (15, 15), 5) 
        self._plot(fig, 4, gimg, 'Guassian Filtering')
        self._hide_ticks(fig)
        return fig
    
    def sharpening(self, **kwargs): #image processing
        title = kwargs.get('title', 'Blurring')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(15, 5))
        fig.subplots_adjust(hspace=.1, wspace=.03)
        fig.suptitle( title, fontsize=25, y=1.09)
        #core
        self._plot2(fig, 1, img, 'original')
        gimg = cv2.GaussianBlur(img, (0, 0), 15) 
        uimg = cv2.addWeighted(img, 1.3, gimg, -.3, 0)
        self._plot2(fig, 2, uimg, 'Unsharp masking')
        gimg = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
        uimg = cv2.addWeighted(img, 1.1, gimg, -.1, 0)
        self._plot2(fig, 3, uimg, 'Laplacian masking')
        self._hide_ticks(fig)
        return fig
    
    def _optimize_shape(self, img):
        rows, cols = img.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        img = cv2.copyMakeBorder(img, 0, nrows-rows, 0, ncols-cols, cv2.BORDER_CONSTANT, value = 0)
        return img
    
    def disk_mask(self, cx=5, cy=5, n=11, r=3, fill=1):
        y,x = np.ogrid[-cx:n-cx, -cy:n-cy]
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n))
        array[mask] = fill
        return array

    def make_odd(self, val):
        val = val if (val % 2 ==1) else (val - 1)
        return val
    
    def squre_mask(self, img, sigma=20, inverse=0):
        hs, ws = img.shape
        hms, wms = hs/2, ws/2
        sqsize = self.make_odd(min(hs, ws))
        xshift = ( ws - sqsize )/2
        yshift = ( hs - sqsize )/2
        x = self.disk_mask( sqsize/2, sqsize/2, sqsize, sigma)
        x = ( 1 - x ) if inverse else x
        mask = np.ones(( hs, ws, 2),np.uint8) if inverse else np.zeros(( hs, ws, 2),np.uint8)
        mask[ yshift:sqsize+yshift, xshift:sqsize+xshift, 0] = x
        mask[ yshift:sqsize+yshift, xshift:sqsize+xshift, 1] = x
        return mask
    
    def gaussian_mask(self, img, sigma=1, inverse=False):
        hs, ws = img.shape
        sqsize = self.make_odd(min(hs, ws))
        xshift = ( ws - sqsize )/2
        yshift = ( hs - sqsize )/2
        x = cv2.getGaussianKernel( sqsize, sigma )
        gaussian = ( x*x.T ) * 1000
        gaussian = ( 1 - gaussian ) if inverse else gaussian
        mask = np.ones(( hs, ws, 2),np.float16) if inverse else np.zeros(( hs, ws, 2),np.float16)
        mask[ yshift:sqsize+yshift, xshift:sqsize+xshift, 0] = gaussian
        mask[ yshift:sqsize+yshift, xshift:sqsize+xshift, 1] = gaussian
        return mask
    
    def butterworth_mask(self, img, sigma=1):
        rows, cols = img.shape
        crow, ccol = rows/2 , cols/2
        sqsize = min(rows, cols)
        x = cv2.getGaussianKernel( sqsize, sigma)
        gaussian = x*x.T
        mask = np.zeros((rows,cols,2),np.float16)
        mask[:sqsize, :sqsize, 0] = gaussian
        mask[:sqsize, :sqsize, 1] = gaussian
        return mask
    
    def _plot3(self, fig, index, title, img):
        ax = fig.add_subplot(1, 4, index)
        ax.imshow(img, cmap = 'gray')
        ax.set_title(title), ax.xaxis.set_ticks([]), ax.yaxis.set_ticks([])

    def frequency_filtering(self, **kwargs):
        #title = kwargs.get('title', 'Freuency filtering')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(16, 5))
        fig.subplots_adjust(hspace=.03, wspace=.03)
        #fig.suptitle( title, fontsize=24, y=1)
        #core
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = self._optimize_shape(img)
        dft = cv2.dft( np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        _mask = kwargs.get('mask', self.squre_mask(img, 0.05))
        fshift = dft_shift*_mask
        f_ishift = np.fft.ifftshift(fshift)
        img_idft = cv2.idft(f_ishift)
        img_idft = cv2.magnitude( img_idft[:,:,0], img_idft[:,:,1])
        self._plot3(fig, 1, 'Original', img)
        self._plot3(fig, 2, 'Frequency Spectrum', magnitude_spectrum)
        self._plot3(fig, 3, 'Mask', 1- _mask[:,:,0])
        self._plot3(fig, 4, 'Result', img_idft)
        return fig

# <codecell>

if __name__ == '__main__':
  cp = CrazyPlotter()
  cp.blurring(title='Blurring').show()

# <markdowncell>

# import matplotlib.pyplot as plt
# 
# class Plotter(object):
#     def __init__(self, xval=None, yval=None):
#         self.xval = xval
#         self.yval = yval
# 
#     def plotthing(self):
#         f = plt.figure(
#         sp = f.add_subplot(111)
#         sp.plot(self.xval, self.yval, 'o-')
#         return f
#     
# app = Plotter(xval=range(0,10), yval=range(0,10))
# plot = app.plotthing()
# plot.show()
# #raw_input()

