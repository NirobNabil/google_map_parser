from skimage.morphology import skeletonize
import sys
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
    self.drones = []
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

  global nodes
  nodes = [ [None for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
  processed = [ [False for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
  bigNodes = []
  node_list = []
  for y in range(0,len(skeleton)):
    for x in range(0,len(skeleton[0])):
      if skeleton[y][x]:
        nodes[y][x] = Node(x,y)
        gg = get_connected(x,y,skeleton)
        # if len(gg)>2:
        #   bigNodes.append((x,y))
        for node in gg:
          nodes[y][x].addNode(node[0],node[1])
        node_list.append(nodes[y][x])

  img = skeleton.copy().astype(int)
  image = [[[0,0,0] for y in range(len(img[0]))] for y in range(len(img))]


  queue = deque()
  def dfs(node, i, im):
    processed[node.y][node.x] = True
    if i == im:
      # print("xd")
      image[node.y][node.x] = [0,255,0]
      i=0
    # print(node.x, node.y, ' - ', node.connected)
    for x,y in node.connected:
      # print(x,y)
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
    # print('nn', g)
    dfs(g[0], g[1], 5)

  bigNodes_dict = {}
  bigEdges = []
  # processed = [ [False for x in range(0,len(nodes[0]))] for y in range(0,len(nodes)) ]
  # queue = deque()
  # def dfs2(node, lastBig, weight):
  #   global cn 
  #   cn = node
  #   if (len(node.connected)>2 or len(node.connected)==1):
  #     bigNodes.append((node.x,node.y))
  #     bigNodes_dict[(node.x,node.y)] = True
  #   if not processed[node.y][node.x]:
  #     for x,y in node.connected:
  #       queue.append( ((x,y), (lastBig.x,lastBig.y), weight+1) )
    
  #   processed[node.y][node.x] = True

  # t = nodes[44][9]
  # queue.append( ((t.x,t.y), (t.x,t.y), 0) )   #this got a little bug
  # while(len(queue)):
  #   g = queue.pop()
  #   # print(g)
  #   dfs2(nodes[g[0][1]][g[0][0]], nodes[g[1][1]][g[1][0]], g[2])
  
  processed = [ [False for x in range(0,len(nodes[0]))] for y in range(0,len(nodes)) ]
  # sys.setrecursionlimit(10**8)
  q = deque()
  def dfs3(node):
    processed[node.y][node.x] = True
    if(len(node.connected)!=2):
      bigNodes.append((node.x,node.y))
      # bigNodes_dict[(node.x,node.y)] = True
    for x,y in node.connected:
      if not processed[y][x]:
        q.append(nodes[y][x])

  q.append(node_list[0])
  while(len(q)):
    dfs3(q.pop())

  def follow_until_bignode(node, parent, weight):
    if(len(node.connected)!=2):
      return [node, weight]
    for x,y in node.connected:
      if parent.x != x or parent.y != y:
        return follow_until_bignode(nodes[y][x], node, weight+1)

  processed = [ [False for x in range(0,len(nodes[0]))] for y in range(0,len(nodes)) ]
  for binod in bigNodes:
    bigNode = nodes[binod[1]][binod[0]]
    for x,y in bigNode.connected:
      n,w = follow_until_bignode(nodes[y][x], bigNode, 1)
      bigEdges.append((binod, (n.x,n.y), w))
    

  intersections = list(set(bigNodes))
  # bigNodes = intersections
  # print_neighbour(41,104,5,5,image)
  for point1 in intersections:
      for point2 in intersections:
        if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 3**2) and (point1 != point2):
          intersections.remove(point2)
  for x,y in intersections:
    image[y][x] = [255,0,0]
  plt.tight_layout()
  # plt.imshow(image)
  # plt.show()
      
  g = list(map(lambda x:x.dictified(), node_list))
  l = dict(node_list=g, big_nodes=bigNodes, big_edges=bigEdges)
  a = json.dumps(l)
  file = open('gg.txt', 'w')
  file.write(a)

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



######### your data starts here. dont concern yourself with anything anything above this line
def read_data():
  file = open('/home/twin_n/workspace/dronet/map_generator/gg.txt', 'r')
  content = file.read()
  return json.loads(content)



data = read_data()
node_list = data['node_list']  #list of objects (instances of Node class)
temp = {}
for node in node_list:
  t_x = node['x']
  t_y = node['y']
  try:
    temp[t_y][t_x] = node
  except KeyError:
    temp[t_y] = {}
    temp[t_y][t_x] = node
node_matrix = temp    # format - node_matrix[y][x] = node

bigNodes = data['big_nodes']   #list of coordinates as arrays [x,y]
bigEdges = data['big_edges']   #list of [[x1,y1], [x2,y2], w]
temp = {}
for n1, n2, w in bigEdges:
  t_n1 = (n1[0], n1[1])
  t_n2 = (n2[0], n2[1])
  try:
    temp[t_n1][t_n2] = w
  except KeyError:
    temp[t_n1] = {}
    temp[t_n1][t_n2] = w
  try:
    temp[t_n2][t_n1] = w
  except KeyError:
    temp[t_n2] = {}
    temp[t_n2][t_n1] = w
bigEdges = temp   # format - bigEdges[y][x] = w

class Drone:
  '''contains drone's attribute data'''
  def __init__(self, capacity, speed, duration, node=None, edge=None):
    self.capacity = capacity
    self.speed = speed
    self.duration = duration
    self.location = {node: node, edge: edge}

drones = [ Drone(50, 30, 300), Drone(50, 10, 600), Drone(80, 6, 100), Drone(30, 10, 500), ]


class Station:
  '''contains drone's attribute data'''
  def __init__(self, capacity, location):
    self.capacity = capacity
    self.location = location  #location is a node object 

stations = [ Station(500, bigNodes[5]), Station(300, bigNodes[8]), Station(1000, bigNodes[40]) ]

from skimage import data, io
def showdebug():
  blobs = invert(data.img_as_bool(imread('map.png', as_gray=1)))
  skel, distance = medial_axis(blobs, return_distance=True)
  skeleton = skeletonize(blobs)
  show(skeleton)
  

def debug():
  r = []
  for i in bigEdges.keys():
    if(len(bigEdges[i].keys())):
      r.append(i)
      r.append(bigEdges[i])
  return r

def check():
  for x in bigEdges.keys():
    if bigNodes.count([x[0], x[1]]) == 0:
      return False
    for y in bigEdges[x].keys():
      if bigNodes.count([y[0], y[1]]) == 0:
        return False
  return True
# def add_request(node):

##  when you get a request from node n, detect which which bigEdge node n is on. say node n is on the 
##  edge between bignodex and bignodey. then add two edges (bignodex,n,calculated_weight_from_bignodex_to_n) and 
##  (bignodey,n,calculated_weight_from_bignodey_to_n) to bigEdges and continue the algorithm
##  after the delivery is done remove these two edges


# je point e request ashse oi point er closest je route ache oi drone ta n1 node er request ta accept korlo je package ta n2 te deliver korte hobe and drone ta delivery nawar age d2 te jacchilo. 

# int calc_new_delivery_cost(n1, n2, prev_cost, max_cost){
#   d1 = closest drone current position
#   d2 = closest drone current delivery destination
#   d_cost_prev = closest drone current delivery cost
#   d_cost_no_ch = d_cost_after = shortest_path(d1,n1,n2,d2)
#   max_cost = min(d_cost_no_ch+prev_cost, max_cost)
#   for every closest charging station near n1 as c_i:
#     s_path = shortest_path(d1,n1,c_i,d2)
#     if s_path < d_cost_no_ch and prev_cost+s_path < max_cost:
#     d_cost_after = min(d_cost_after, s_path + calc_new_delivery_cost(c_i, n2, prev_cost+s_path, max_cost) )

# def get_closest_drone(node):


# def calc_nd_cost(n1,n2,prev_cost,max_cost):
#   d1 = 


#   return d_cost_after - d_cost_prev;
# }