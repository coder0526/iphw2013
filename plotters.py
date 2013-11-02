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
                
    def _hide_ticks(self, fig):
        for i, ax in enumerate(fig.axes):
            #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            #ax.set_title('good%d' % i)
            #ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            
    def _hist_plot(self, axis, img):
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
                
            
    def _sub_plotting(self, fig, index, img, title='untitled'):
        grid = GridSpec(4, 3)
        ax1 = fig.add_subplot(grid[:3, index])
        ax1.set_title(title)
        ax1.imshow(img, cmap="Greys_r")
        ax2 = fig.add_subplot(grid[3:, index])
        ax2.set_title(title+'-histogram')
        self._hist_plot(ax2, img)
        ax2.set_xlim(0, 256)
            
    def demo_ip(self, **kwargs): #image processing
        title = kwargs.get('title', 'untitled')
        img = kwargs.get('img', None)
        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(hspace=.14, wspace=.03)
        fig.suptitle( title, fontsize=25, y=.96)
        self._sub_plotting(fig, 0, img, 'original')
        gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self._sub_plotting(fig, 1, gimg, 'grayed')
        egimg = cv2.equalizeHist(gimg)
        self._sub_plotting(fig, 2, egimg, 'equalized')
        self._hide_ticks(fig)
        return fig
        #plt.show()

# <codecell>

if __name__ == '__main__':
  cp = CrazyPlotter()
  cp.demo_ip(title='Scarlet Histograms').show()

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

# <codecell>


