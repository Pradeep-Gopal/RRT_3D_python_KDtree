import cv2
import numpy as np
import time
import math
import random 
from sklearn.neighbors import KDTree
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import spatial
from queue import PriorityQueue 
q = PriorityQueue() 
plt.ion()

length = 10
breadth = 10
height = 10
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim3d(0, 10)
ax.set_ylim3d(0,10)
ax.set_zlim3d(0,10)

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
    new_point = (j[0]+3*vec[0], j[1]+ 3*vec[1], j[2]+3*vec[2])
    return new_point

def neighbours(seed, r, tree):
    results = tree.query_ball_point((seed), r)
#     print(results)
    nearby_points = X[results]
#     print("radius = ", r)
#     print("nearby_points", nearby_points)
    return nearby_points
        


start = (5, 5, 5) 
goal_x, goal_y, goal_z = (9.5,9.5,9.5)
visited_nodes = set()
all_nodes = []
parent_list = []
seed_list = []

visited_nodes.add(start)
all_nodes.append(start)

parent_dict = {}
cost_dict = {}
parent_dict[start] = "okay"
cost_dict[start] = 0

seed = (0,0,0)
print("visitednodes",visited_nodes)
print("all_nodes", all_nodes)
print("\n")

i = 0

while(goalcheck_circle(goal_x, goal_y, goal_z, seed[0], seed[1], seed[2]) == False):
#     if(i ==5):
#         break
    
#     seed = seeds[i]
#     i = i+1
#     print("\n")
#     print("seeed = ", seed)
    seed = generate_seed()

    if seed not in visited_nodes:
        
#         all_nodes.insert(0, seed) 
        X = np.asarray(all_nodes)
        tree = spatial.KDTree(X)
        
        r = 1.5
        n = (0,0,0)

        while(1):
            n = neighbours(seed, r,tree)
#             print("\n")
            if(n == seed).all():
                r = r + 1
            else:
                break
#         print("neighbours", n)
        
        
        for pt in n:
#             print("pt", pt)
            pt = tuple(pt)
            cost = cost_dict[pt]
            cost_new = cost + cost2go(pt, seed)
#             print("cost_new", cost_new)
            q.put((cost_new, pt, cost))
        
        parent = q.get()
        q = PriorityQueue() 
        parent = parent[1] 
#         print("selected_neighbour", parent)
              
        
        if (cost2go(parent,seed) > 3):
            seed = max_step_prop(parent, seed)
            seed = (round(seed[0], 1), round(seed[1], 1), round(seed[2], 1))
            
        all_nodes.insert(0, seed)
#         print("f_parent", parent)
#         print("f_seed", seed)
            
            
        visited_nodes.add(seed)
        parent_dict[seed] = parent 
        neww_cost = cost2go(seed, parent) + cost_dict[parent]
        
#         print("final_cost", neww_cost)
        cost_dict[seed] = neww_cost
        
        parent_list.append((parent[0], parent[1], parent[2]))
        seed_list.append((seed[0], seed[1], seed[2]))
        ax.plot3D((parent[0],seed[0]), (parent[1], seed[1]), (parent[2], seed[2]), 'black')
        # plt.show()
#         print("\n")

# print("\n")
# print("final parent dict", parent_dict)
# print("\n")
# print("points", all_nodes)
# print("\n")
# print("visited_nodes", visited_nodes)
# print("\n")
# print("cost_dict", cost_dict)
# print("\n")
print("Goal_Reached")


path = []
path.append((goal_x, goal_y, goal_z))

while parent != "okay":
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
plt.savefig("RRT_star.png")
plt.pause(5)
plt.close()

