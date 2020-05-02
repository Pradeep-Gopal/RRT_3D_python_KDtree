import cv2
import numpy as np
import time
import math
import random 
from sklearn.neighbors import KDTree
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.ion()

length = 10
breadth = 10
height = 10
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim3d(0, 11)
ax.set_ylim3d(0,11)
ax.set_zlim3d(0,11)

def boundary_check(i, j, k):
    if (i < 0) or (j < 0) or (k < 0) or (i >= length) or (j >= breadth) or (k >= height):
        return True
    else:
        return False

def generate_seed():
    x = round(random.uniform(0 , length)*2)/2
    y = round(random.uniform(0 , breadth)*2)/2
    z = round(random.uniform(0 , height)*2)/2
#     print(x,y,z)
    return (x,y,z)
    
def goalcheck_circle(x, y, z, goal_x, goal_y, goal_z):
    if ((x - goal_x) ** 2 + (y - goal_y) ** 2 + (z - goal_z) **2 <= (0.5 ** 2)):
        return True
    else:
        return False

def cost2go(pt1, pt2):
    dist = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2) 
    return dist

def max_step_prop(j, i):
    k = (i[0] - j[0], i[1] - j[1], i[2] - j[2])
    k_mod = math.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    vec = (k[0]/k_mod, k[1]/k_mod, k[2]/k_mod)
    new_point = (j[0]+1*vec[0], j[1]+ 1*vec[1], j[2]+1*vec[2])
    return new_point


start = (1,1,1)
goal_x, goal_y, goal_z = (9,9,9)
visited_nodes = set()
all_nodes = []
parent_list = []
seed_list = []

visited_nodes.add(start)
all_nodes.append(start)

parent_dict = {}
parent_dict[start] = None

seed = (0,0,0)
print("visitednodes",visited_nodes)
print("all_nodes", all_nodes)
print("\n")

while(goalcheck_circle(goal_x, goal_y, goal_z, seed[0], seed[1], seed[2]) == False):
    seed = generate_seed()
    print("generated_seed", seed)
    if seed not in visited_nodes:
        
        all_nodes.insert(0, seed)    
        X = np.array(all_nodes)
        tree = KDTree(X, leaf_size=2) 
        dist, ind = tree.query(X[:1], k=2)  
        p = ind[0][1]
        parent = all_nodes[p]


        if (cost2go(parent,seed) > 1):
            seed = max_step_prop(parent, seed)
            seed = (round(seed[0], 1), round(seed[1], 1), round(seed[2], 1))
            all_nodes[0] = seed
            visited_nodes.add(seed)
        else:
            visited_nodes.add(seed)

        parent_dict[seed] = parent 
        print("parent", parent)
        print("seed", seed)

        parent_list.append((parent[0], parent[1], parent[2]))
        seed_list.append((seed[0], seed[1], seed[2]))
        ax.plot3D((parent[0],seed[0]), (parent[1], seed[1]), (parent[2], seed[2]), 'black')
        # plt.show()
        print("\n")


print("Goal_Reached")

path = []
path.append((goal_x, goal_y, goal_z))

while parent != None:
    temp = parent_dict.get(parent)
    path.append(parent)
    parent = temp
    if parent == (start):
        break

path.append(start)
print("Backtracking done - shortest path found")

path = path[::-1]


x_path = [path[i][0] for i in range(len(path))]
y_path = [path[i][1] for i in range(len(path))]
z_path = [path[i][2] for i in range(len(path))]


# l = 0

# while l < len(seed_list):
#             # ax.plot3D((parent[0],seed[0]), (parent[1], seed[1]), (parent[2], seed[2]), 'black')
#     ax.plot3D((parent_list[l][0], seed_list[l][0]), (parent_list[l][1], seed_list[l][1]), (parent_list[l][2], seed_list[l][2]),  'black')
#     l = l + 1
#     plt.show()
#     plt.pause(0.000000000000000000000000000000000001)

ax.plot3D(x_path, y_path, z_path, "-r")
print(path)
plt.show()
plt.savefig("output.png")
plt.pause(5)
plt.close()

        
        
    
    
