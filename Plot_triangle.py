
"""
This Python script plots TRIANGLE files into a visualisable plot.
"""

#Set up dis

#Set up display environment in putty
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


#----------------------------------------
# This one reads the geometry of the nodes
def Read_node (Input):
    f = open(Input + '.node', 'r')
    lines = f.readlines()
    Header = lines[0].split()
    index = 0
    for i in Header:
        Header[index] = int(i)
        index = index+1

    Nnod = Header[0]; Ndim = Header[1]; Nattr = Header[2]; Nmark = Header[3]
    Data = np.zeros((Nnod, 1+Ndim+Nattr+Nmark), dtype = np.float)

    for l_index in range(1,Nnod+1):
        Columns = lines[l_index].split()
        for c_index in range(0,1+Ndim+Nattr+Nmark):
            Data[l_index-1,c_index] = float(Columns[c_index])

    return Data




#-----------------------------------------------------------
#This one reads the element files
def Read_ele(Input):
    f = open(Input + '.ele', 'r')
    lines = f.readlines()
    Header = lines[0].split()
    index = 0
    for i in Header:
        Header[index] = int(i)
        index = index+1

    Ntri = Header[0]; Nnod = Header[1]; Nattr = Header[2]
    Data = np.zeros((Ntri, 1+Nnod+Nattr), dtype = np.float)

    for l_index in range(1,Ntri+1):
        Columns = lines[l_index].split()
        for c_index in range(0,1+Nnod+Nattr):
            Data[l_index-1,c_index] = float(Columns[c_index])

    return Data




#----------------------------------------
# This one reads the geometry of the nodes
def Read_poly (Input):
    f = open(Input + '.poly', 'r')
    lines = f.readlines()

    #first section: nodes
    Header_nodes = lines[0].split()
    index = 0
    for i in Header_nodes:
        Header_nodes[index] = int(i)
        index = index+1

    Nnod = Header_nodes[0]; Ndim = Header_nodes[1]; Nattr = Header_nodes[2]; Nmark = Header_nodes[3]
    Nodes = np.zeros((Nnod, 1+Ndim+Nattr+Nmark), dtype = np.float)

    for l_index in range(1,Nnod+1):
        Columns = lines[l_index].split()
        for c_index in range(0,1+Ndim+Nattr+Nmark):
            Nodes[l_index-1,c_index] = float(Columns[c_index])


    # Second section: segments
    Header_seg = lines[Nnod+1].split()
    index = 0
    for i in Header_seg:
        Header_seg[index] = int(i)
        index = index+1

    Nnod = Header_seg[0]; Nmark = Header_seg[1]
    Segments = np.zeros((Nnod,3+Nmark), dtype = np.float)

    for l_index in range(1,Nnod+1):
        Columns = lines[l_index+Nnod+1].split()
        for c_index in range(0,3+Nmark):
            Segments[l_index-1,c_index] = float(Columns[c_index])

    return Nodes, Segments



#--------------------------------------------
# Read the POLY file
Nodes, Segments = Read_poly('A')

Nodes = Read_node('A.1')
Elements = Read_ele('A.1')

print Elements

#-------------------------------------------------------
fig=plt.figure(1, facecolor='White',figsize=[10,10])
ax = plt.subplot2grid((1,2),(0,0),colspan=1, rowspan=1)
ax.set_title('Input .poly file ', fontsize = 22)
ax.set_xlabel('X - Distance from origin (m)', fontsize = 18)
ax.set_ylabel('Y - Distance from origin (m)', fontsize = 18)

# Draw the nodes
ax.scatter(Nodes[:,1], Nodes[:,2], c = Nodes[:,3], s= 50, linewidth = 0, alpha = 0.5)
# Draw the Segments
for i in range(len(Segments)):
    Ogn_num = Segments[i,1]
    Dst_num = Segments[i,2]

    Ogn_index = np.where(Nodes[:,0]==Ogn_num); Ogn_index = Ogn_index[0][0]
    Dst_index = np.where(Nodes[:,0]==Dst_num); Dst_index = Dst_index[0][0]

    X_coord = np.array([Nodes[Ogn_index,1],Nodes[Dst_index,1]])
    Y_coord = np.array([Nodes[Ogn_index,2],Nodes[Dst_index,2]])

    ax.plot(X_coord, Y_coord, color = plt.cm.jet(i*10))


ax = plt.subplot2grid((1,2),(0,1),colspan=1, rowspan=1)
ax.set_title('Output .node and .ele files ', fontsize = 22)
ax.set_xlabel('X - Distance from origin (m)', fontsize = 18)
ax.set_ylabel('Y - Distance from origin (m)', fontsize = 18)

# Draw the nodes
ax.scatter(Nodes[:,1], Nodes[:,2], c = Nodes[:,3], s= 50, linewidth = 0, alpha = 0.5)


# Draw the Segments
for i in range(len(Segments)):
    Ogn_num = Segments[i,1]
    Dst_num = Segments[i,2]

    Ogn_index = np.where(Nodes[:,0]==Ogn_num); Ogn_index = Ogn_index[0][0]
    Dst_index = np.where(Nodes[:,0]==Dst_num); Dst_index = Dst_index[0][0]

    X_coord = np.array([Nodes[Ogn_index,1],Nodes[Dst_index,1]])
    Y_coord = np.array([Nodes[Ogn_index,2],Nodes[Dst_index,2]])

    ax.plot(X_coord, Y_coord, 'k')

# Draw the Element boundaries
for i in range(len(Elements)):
    Pt1_num = Elements[i,1]
    Pt2_num = Elements[i,2]
    Pt3_num = Elements[i,3]

    Pt1_index = np.where(Nodes[:,0]==Pt1_num); Pt1_index = Pt1_index[0][0]
    Pt2_index = np.where(Nodes[:,0]==Pt2_num); Pt2_index = Pt2_index[0][0]
    Pt3_index = np.where(Nodes[:,0]==Pt3_num); Pt3_index = Pt3_index[0][0]


    X_coord = np.array([Nodes[Pt1_index,1],Nodes[Pt2_index,1],Nodes[Pt3_index,1]])
    Y_coord = np.array([Nodes[Pt1_index,2],Nodes[Pt2_index,2],Nodes[Pt3_index,2]])

    ax.plot(X_coord, Y_coord, '--k')# , color = plt.cm.jet(i*10))



plt.savefig('AAA.png')
