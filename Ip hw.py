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
kelly = read_img_from('http://www.hollywoodreporter.com/sites/default/files/imagecache/blog_post_349_width/2012/12/kelly-clarkson-pr-2012-p.jpg')

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
cp.blurring(title='Taylor Swift Blurring', img=taylor).show()

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
cp.sharpening(title='Kelly Clarkson Sharpening', img=kelly).show()

# <markdowncell>

# ##2-3. Frequency domain filtering
#   - Apply ILPF, BLPF, and GLPF to an input image
#   - Apply IHPF, BHPF, and GHPF to an input image

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
for i in range(3, 10, 3):
  cp.frequency_filtering( img=taylor, mask=cp.squre_mask(img, 5*i, 0)).show()
for i in range(3, 10, 3):
  cp.frequency_filtering( img=taylor, mask=cp.squre_mask(img, 5*i, 1)).show()

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
for i in range(3, 10, 3):
    cp.frequency_filtering( img=taylor, mask=cp.gaussian_mask(img, 3*i, 0)).show()   
for i in range(3, 10, 3):    
    cp.frequency_filtering( img=taylor, mask=cp.gaussian_mask(img, 3*i, inverse=True)).show()

# <codecell>

from plotters import CrazyPlotter as crazy
reload(sys.modules['plotters'])
cp = crazy()
for i in range(1, 4):
    cp.frequency_filtering( img=taylor, mask=cp.butterworth_mask(img, .02*i, inverse=0)).show()
for i in range(1, 4):
    cp.frequency_filtering( img=taylor, mask=cp.butterworth_mask(img, .02*i, inverse=1)).show()

# <codecell>

# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))

# creating a guassian filter
x = cv2.getGaussianKernel( 20, 2)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()

# <codecell>

size = 60
aa = np.ndarray( ( size, size))
#mask= cp.gaussian_mask( aa, 15, 1)
plt.imshow( mask[:,:,0], cmap = 'gray')
msize= int(size/2) + 20
mwid = min(int(size/8), 10)
for x in range( msize- mwid, msize+mwid, 1):
    r = ""
    for y in range( msize-mwid, msize+mwid, 1):
        r += "_%.4f" % mask[x,y,0]
    print r
img.shape

# <codecell>

from scipy.signal import butter, lfilter
 
def butter_bandpass_filter( lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a
    #y = lfilter(b, a, data)
    #return y

# <codecell>

b,a = butter_bandpass_filter(3, 10, .6, order= 5)

