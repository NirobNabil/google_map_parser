from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from skimage.morphology import skeletonize
from skimage import data, io
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.viewer import ImageViewer
from skimage.io import imread
import numpy as np
from collections import deque
import json
from skimage.morphology import medial_axis, skeletonize
import os

class Node:
  '''this is the common node obj'''
  def __init__(self,x=-1,y=-1):
    self.x = x
    self.y = y
    self.connected = []
  def __str__(self):
    return str(self.x) + " " + str(self.y) + ' - ' + str(self.connected)
  def addNode(self,x,y):
    self.connected.append((x,y))
  def dictified(self):
    x = dict(x=self.x, y=self.y, connected=self.connected)
    return x

def get_connected(x,y,skeleton):
  res = []
  for yi in [-1,0,1]:
    for xi in [-1,0,1]:
      try:
        if skeleton[y+yi][x+xi] and (xi!=0 or yi!=0) and x+xi>=0 and y+yi>=0:
          res.append((x+xi,y+yi))
      except IndexError:
        pass

  # print(res)
  def immediate(i_x,i_y):
    r = []
    for xi,yi in [(0,-1),(0,1),(-1,0),(1,0)]:
      try:
        # print(i_x+xi,',',i_y+yi,'-',skeleton[i_y+yi][i_x+xi])
        if skeleton[i_y+yi][i_x+xi] and (i_x+xi!=x or i_y+yi!=y):
          r.append((i_x+xi,i_y+yi))
      except IndexError:
        pass
    return r

  for x1,y1 in immediate(x,y):
    # if skeleton[x1,y1]:
    # print("for", x1,',',y1)
    for x2,y2 in immediate(x1,y1):
      try:
        res.remove((x2,y2))
      except ValueError:
        pass
  
  # print(res)

  return res

nodes = []
processed = []
bigNodes = []
node_list = []
def generate():
  blobs = invert(data.img_as_bool(imread('map.png', as_gray=1)))

  # Compute the medial axis (skeleton) and the distance transform
  skel, distance = medial_axis(blobs, return_distance=True)

  # Compare with other skeletonization algorithms
  skeleton = skeletonize(blobs)
  skeleton_lee = skeletonize(blobs, method='lee')

  # Distance to the background for pixels of the skeleton
  dist_on_skel = distance

  nodes = [ [None for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
  processed = [ [False for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
  for y in range(0,len(skeleton)):
    for x in range(0,len(skeleton[0])):
      if skeleton[y][x]:
        nodes[y][x] = Node(x,y)
        gg = get_connected(x,y,skeleton)
        if len(gg)>2:
          bigNodes.append((x,y))
        for node in gg:
          if x == 249 and y == 165:
            print(x,y, '-', node[0], node[1])
          nodes[y][x].addNode(node[0],node[1])
        node_list.append(nodes[y][x])

  img = skeleton.copy().astype(int)
  image = [[[0,0,0] for y in range(len(img[0]))] for y in range(len(img))]


  queue = deque()
  def dfs(node, i, im):
    processed[node.y][node.x] = True
    if i == im:
      print("xd")
      image[node.y][node.x] = [0,255,0]
      i=0
    print(node.x, node.y, ' - ', node.connected)
    for x,y in node.connected:
      print(x,y)
      if processed[y][x]:
        pass
      else:
        queue.append( (nodes[y][x], i+1))
      # dfs(nodes[y][x], i+1, im)

  queue.append((node_list[0], 0))
  # dfs(node_list[0], 0, 3)
  g = (node_list[0],0)
  while len(queue):
    g = queue.pop()
    print('nn', g)
    dfs(g[0], g[1], 5)


  intersections = list(set(bigNodes))
  bigNodes = intersections
  # print_neighbour(41,104,5,5,image)
  for point1 in intersections:
      for point2 in intersections:
        if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 3**2) and (point1 != point2):
          intersections.remove(point2)
  for x,y in intersections:
    image[y][x] = [255,0,0]
  plt.tight_layout()
  plt.imshow(image)
  plt.show()
      
  g = list(map(lambda x:x.dictified(), node_list))
  l = dict(node_list=g, big_nodes=bigNodes)
  a = json.dumps(l)
  file = open('gg.txt', 'w')
  file.write(a)

def read_data():
  file = open('/home/twin_n/workspace/dronet/map_generator/server/dronet/gg.txt', 'r')
  content = file.read()
  return json.loads(content)

def print_neighbour(x,y,dx,dy,skeleton):
  r = []
  a = []
  lx = len(skeleton[0])
  ly = len(skeleton)
  for i in range(-dy,dy+1):
    b = []
    # for ix in range(x-dx+1,x+dx+2):
    #   b.append(ix)
    # a.append(b)
    r.append(skeleton[y+i][x-dx:x+dx+1])
    # print(skeleton[y+i][x-dx:x+dx].astype(int))
  print(np.array(a))
  plt.imshow(r,cmap='magma')
  plt.show()

def show(skeleton):
  plt.imshow(skeleton, cmap=plt.cm.gray)
  plt.show()

@api_view(['GET'])
def drone_data(request):
  data = read_data()
  node_list = data['node_list']
  bigNodes = data['big_nodes']
  bigEdges = data['big_edges']
  return Response({'node_list':node_list, 'bigNodes':bigNodes, 'bigEdges':bigEdges })