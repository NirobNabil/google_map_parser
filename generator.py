from skimage.morphology import skeletonize
from skimage import data, io
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.viewer import ImageViewer
from skimage.io import imread
import numpy as np
from collections import deque
# import cv2
from skimage.morphology import medial_axis, skeletonize
# Invert the horse image
# image = invert(data.horse())
blobs = invert(data.img_as_bool(imread('map_debug.png', as_gray=1)))

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(blobs, return_distance=True)

# Compare with other skeletonization algorithms
skeleton = skeletonize(blobs)
skeleton_lee = skeletonize(blobs, method='lee')

# Distance to the background for pixels of the skeleton
dist_on_skel = distance

class Node:
  '''this is the common node obj'''
  def __init__(self,x=None,y=None):
    self.x = x
    self.y = y
    self.connected = []
  def __str__(self):
    return str(self.x) + " " + str(self.y) + ' - ' + str(self.connected)
  def addNode(self,x,y):
    self.connected.append((x,y))


def get_connected(x,y,skeleton):
  res = []
  for yi in [-1,0,1]:
    for xi in [-1,0,1]:
      try:
        if skeleton[y+yi][x+xi] and (xi!=0 or yi!=0):
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

nodes = [ [None for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
processed = [ [False for x in range(0,len(skeleton[0]))] for y in range(0,len(skeleton)) ]
bigNodes = []
node_list = []
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
# image[:len(image-1)][:len(image[0]-1)] = 0
# for y in range(0,len(nodes)-1):
#   for x in range(0,len(nodes[0])-1):
#     if img[y][x]==1:
#       image[y][x] = [255,255,255]
#     else: 
#       image[y][x] = [0,0,0]
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

# for node in node_list:
#   image[node.y][node.x] = [0,255,0]

intersections = list(set(bigNodes))
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
  

# def gg():  
#   sk = skeleton.copy()
#   for y in range(len(skeleton)-1):
#     for x in range(len(skeleton[0])-1):
#       if skeleton[y][x] and check_neighbour(y,x, skeleton)>2:
#         print(x,',',y)
#         sk[y][x] = 0
#   plt.imshow(sk,cmap=plt.cm.gray)
#   plt.show()
# plt.show()
# gg()

def show(skeleton):
  plt.imshow(skeleton, cmap=plt.cm.gray)
  plt.show()


# print_neighbour(150,130,25,10,skeleton)

# def neighbours_f(x,y,image):
#   """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
#   img = image
#   x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
#   return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   


# def getSkeletonIntersection(skeleton):
#   """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

#   Keyword arguments:
#   skeleton -- the skeletonised image to detect the intersections of

#   Returns: 
#   List of 2-tuples (x,y) containing the intersection coordinates
#   """
#   # A biiiiiig list of valid intersections             2 3 4
#   # These are in the format shown to the right         1 C 5
#   #                                                    8 7 6 
#   validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
#                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
#                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
#                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
#                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
#                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
#                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
#                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
#                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
#                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
#                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
#                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
#                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
#                         [1,0,1,1,0,1,0,0]]
#   image = skeleton.copy()
#   # image = image/255;
#   intersections = list()
#   for x in range(1,len(image)-1):
#     for y in range(1,len(image[x])-1):
#       # If we have a white pixel
#       if image[x][y] == 1:
#         neighbours = neighbours_f(x,y,image)
#         valid = True
#         if neighbours in validIntersection:
#           intersections.append((y,x))
#   # Filter intersections to make sure we don't count them twice or ones that are very close together
#   for point1 in intersections:
#     for point2 in intersections:
#       if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 5**2) and (point1 != point2):
#         intersections.remove(point2)
#   # Remove duplicates
#   image[:len(image-1)][:len(image[0]-1)] = 0
#   intersections = list(set(intersections))
#   for x,y in intersections:
#     image[y][x] = 1
  
#   # print_neighbour(41,104,5,5,image)
#   plt.tight_layout()
#   plt.imshow(image,cmap=plt.cm.gray)
#   plt.show()
#   return intersections

# getSkeletonIntersection(skeleton)


# fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(blobs, cmap=plt.cm.gray)
# ax[0].set_title('original')
# ax[0].axis('off')

# ax[1].imshow(dist_on_skel, cmap='magma')
# ax[1].contour(blobs, [0.5], colors='w')
# ax[1].set_title('medial_axis')
# ax[1].axis('off')

# ax[2].imshow(skeleton, cmap=plt.cm.gray)
# ax[2].set_title('skeletonize')
# ax[2].axis('off')

# ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
# ax[3].set_title("skeletonize (Lee 94)")
# ax[3].axis('off')


# plt.imshow(blobs, cmap=plt.cm.gray)
# plt.show()
# fig.tight_layout()
# plt.show()

