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
        
    def _plot_blur(self, fig, index, img, title='untitled'):
        ax1 = fig.add_subplot(2,2,index)
        ax1.set_title(title)
        if img is not None: ax1.imshow(img, cmap="Greys_r")

    def blurring(self, **kwargs): #image processing
        title = kwargs.get('title', 'Blurring')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=.1, wspace=.03)
        fig.suptitle( title, fontsize=25, y=.96)
        #core
        self._plot_blur(fig, 1, img, 'original')
        bimg = cv2.boxFilter(img, 0, (15, 15))
        self._plot_blur(fig, 2, bimg, 'Averaging(Mean) Filtering')
        mimg = cv2.medianBlur(img, 15) 
        self._plot_blur(fig, 3, mimg, 'Median Filtering')
        gimg = cv2.GaussianBlur(img, (15, 15), 5) 
        self._plot_blur(fig, 4, gimg, 'Guassian Filtering')
        self._hide_ticks(fig)
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

