"""
This Python script makes .poly files to make unstructured maeshes in TRIANGLE
"""

#Set up display environment in putty
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import math as mt
import scipy as sp
from datetime import datetime
from matplotlib import cm
from pylab import *
import matplotlib.ticker as tk
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import *
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools as itt
from osgeo import gdal, osr
from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
from copy import copy
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import cPickle


# Make as functions



Nnod = 30

Ndim = 2

Nattr = 2

Nmark = 1

Header_nodes = [Nnod, Ndim, Nattr, Nmark]

Header_seg = [Nnod, Nmark]




29 2 1 0
1 0.200000 -0.776400 -0.57
2 0.220000 -0.773200 -0.55
3 0.245600 -0.756400 -0.51
4 0.277600 -0.702000 -0.53
5 0.488800 -0.207600 0.28
6 0.504800 -0.207600 0.30
7 0.740800 -0.739600 0
8 0.756000 -0.761200 -0.01
9 0.774400 -0.772400 0
10 0.800000 -0.776400 0.02
11 0.800000 -0.792400 0.01
12 0.579200 -0.792400 -0.21
13 0.579200 -0.776400 -0.2
14 0.621600 -0.771600 -0.15
15 0.633600 -0.762800 -0.13
16 0.639200 -0.744400 -0.1
17 0.620800 -0.684400 -0.06
18 0.587200 -0.604400 -0.01
19 0.360800 -0.604400 -0.24
20 0.319200 -0.706800 -0.39
21 0.312000 -0.739600 -0.43
22 0.318400 -0.761200 -0.44
23 0.334400 -0.771600 -0.44
24 0.371200 -0.776400 -0.41
25 0.371200 -0.792400 -0.42
26 0.374400 -0.570000 -0.2
27 0.574400 -0.570000 0
28 0.473600 -0.330800 0.14
29 0.200000 -0.792400 -0.59
29 0
1 29 1
2 1 2
3 2 3
4 3 4
5 4 5
6 5 6
7 6 7
8 7 8
9 8 9
10 9 10
11 10 11
12 11 12
13 12 13
14 13 14
15 14 15
16 15 16
17 16 17
18 17 18
19 18 19
20 19 20
21 20 21
22 21 22
23 22 23
24 23 24
25 24 25
26 25 29
27 26 27
28 27 28
29 28 26
1
1 0.47 -0.5
