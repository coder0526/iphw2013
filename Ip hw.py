# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
def url_to_array(url):
    request = urllib.urlopen(url)
    arr = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return arr
def read_img_from(url):
  img_arr = url_to_array(url)
  img = cv2.imdecode(img_arr, cv2.CV_LOAD_IMAGE_COLOR)
  img = img[:,:,[2,1,0]] #BGR to RGB
  print "height = %d, width = %d, n_colors = %d" % img.shape
  #print "pixel (300,300) is %s (BGR)" % img[300,300]
  #plt.imshow(img)
  return img
def addplot(pos, title, img):
    plt.subplot(pos[0], pos[1], pos[2])
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="Greys_r")

# <codecell>

scarlet = read_img_from('http://wackymania.com/image/2010/10/most-beatiful-women/most-beatiful-women-08.jpg')
taylor = read_img_from("http://www.bimtri.com/wp-content/uploads/2013/08/Taylor-Swift.jpg")

# <markdowncell>

# ##2-1. Intensity transformation
#   * Apply histogram equalization to an input gray image
#   * Show the histograms of the original image and the result image

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
cp.histo_equal(title='Scarlett Johansson', img=scarlet).show()
#cp.histo_equal(title='Taylor Swift', img=taylor).show()

# <markdowncell>

# ##2-2. Spatial filtering
#   - Apply averaging filters to an input image
#   - Apply median filters to an input image
#   - Apply image sharpening based on Laplacian mask to an input image
#   - Apply unsharp masking to an input imag

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
cp.blurring(title='Taylor Swift', img=taylor).show()

# <markdowncell>

# ##2-3. Frequency domain filtering
#   - Apply ILPF, BLPF, and GLPS to an input image
#   - Apply IHPF, BHPF, and GHPS to an input image

# <codecell>

def twoDconvolution(img):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)    
    showplot(121, 'Source', img)
    showplot(122, 'Result', dst)

# <codecell>

twoDconvolution(taylor)

