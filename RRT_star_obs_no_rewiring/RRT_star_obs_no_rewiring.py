import cv2
import numpy as np
import time
import math
import random 
from sklearn.neighbors import KDTree
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import matplotlib.pyplot as plt 
from matplotlib import style 
from scipy import spatial
import copy
from queue import PriorityQueue 
q = PriorityQueue() 
plt.ion()
# style.use('ggplot') 
length = 10
breadth = 10
height = 10
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlim3d(0, 11)
ax.set_ylim3d(0,11)
ax.set_zlim3d(0,11)

def obstacle_check(i,j,k):
    a = b = c = d = e = f = g = h = 0
    if(2 <= i <=3) and (3 <= j <= 4) and (0 <= k <= 1):
        a = 1
    elif (4 <= i <=5) and (1 <= j <= 2) and (0 <= k <= 3):
        b = 1
    elif (6 <= i <=7) and (5 <= j <= 6) and (0 <= k <= 4):
        c = 1
    elif (8 <= i <=9) and (3 <= j <= 4) and (0 <= k <= 5):
        d = 1
    elif (9 <= i <=10) and (7 <= j <= 8) and (0 <= k <= 10):
        e = 1
    elif (2 <= i <=3) and (7 <= j <= 8) and (0 <= k <= 10):
        f = 1
    elif (5 <= i <=6) and (4 <= j <= 5) and (0 <= k <= 10):
        g = 1
    elif (9 <= i <=10) and (6 <= j <= 7) and (0 <= k <= 10):
        h = 1

    if  ((a == 1) or (b == 1) or (c == 1) or (d == 1) or (e == 1) or (f == 1) or (g == 1) or (h == 1)):
        return True
    else:
        return False

def obstacle_map():
    # defining x, y, z co-ordinates for bar position 
    x = [2,4,6,8,9,2,5,9] 
    y = [3,1,5,3,7,7,4,6] 
    z = np.zeros(8) 

    # size of bars 
    dx = np.ones(8)              # length along x-axis 
    dy = np.ones(8)              # length along y-axs 
    dz = [1,3,4,5,9,9,9,9]   # height of bar 

    # setting color scheme 
    color = [] 
    for h in dz: 
        if h > 5: 
            color.append('b') 
        else: 
            color.append('b') 

    ax.bar3d(x, y, z, dx, dy, dz, color = color) 



def boundary_check(i, j, k):
    if (i < 0) or (j < 0) or (k < 0) or (i >= length) or (j >= breadth) or (k >= height):
        return True
    else:
        return False



def line_obstacle_check(j, i):
    k = (i[0] - j[0], i[1] - j[1], i[2] - j[2])
    k_mod = math.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    vec = (k[0]/k_mod, k[1]/k_mod, k[2]/k_mod)
    new_point = (j[0]+0.1*vec[0], j[1]+ 0.1*vec[1], j[2]+0.1*vec[2])
    return new_point

def cost2go(pt1, pt2):
    dist = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2) 
    return dist


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
    
def max_step_prop(j, i):
    k = (i[0] - j[0], i[1] - j[1], i[2] - j[2])
    k_mod = math.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    vec = (k[0]/k_mod, k[1]/k_mod, k[2]/k_mod)
    new_point = (j[0]+2*vec[0], j[1]+ 2*vec[1], j[2]+2*vec[2])
    return new_point

def neighbours(seed, r, tree):
    results = tree.query_ball_point((seed), r)
#     print(results)
    nearby_points = X[results]
#     print("radius = ", r)
#     print("nearby_points", nearby_points)
    return nearby_points

obstacle_map()
start = (0,0,0)
goal_x, goal_y, goal_z = (9.0,9.0,9.0)
visited_nodes = set()
all_nodes = []
parent_list = []
seed_list = []
parent_dict = {}
cost_dict = {}

visited_nodes.add(start)
all_nodes.append(start)
parent_dict[start] = "okay"
cost_dict[start] = 0

seed = (0,0,0)
# print("visitednodes",visited_nodes)
# print("all_nodes", all_nodes)
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
#     print("generated_seed", seed)
    if ((seed not in visited_nodes) and not obstacle_check(seed[0], seed[1], seed[2])):
        
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
              
        
        if (cost2go(parent,seed) > 2):
            seed = max_step_prop(parent, seed)
            seed = (round(seed[0], 1), round(seed[1], 1), round(seed[2], 1))
            
            
        par = seed
        s = parent
        a = 0
        # print(s)
        while(cost2go(par,s)>=0.1):
            a = line_obstacle_check(s, par)
            # print(a)
            if obstacle_check(a[0], a[1], a[2]):
#                 print("inside")
#                 print("stop point", a)
                break
            s = a

        s = (round(s[0], 1), round(s[1], 1), round(s[2], 1))
         
#         s = seed
        if s not in visited_nodes:
            neww_cost = cost2go(seed, parent) + cost_dict[parent]  
            all_nodes.insert(0, s)
            visited_nodes.add(s)
            parent_dict[s] = parent 
            cost_dict[s] = neww_cost
            parent_list.append((parent[0], parent[1], parent[2]))
            seed_list.append((s[0], s[1], s[2]))
            ax.plot3D((parent[0],s[0]), (parent[1], s[1]), (parent[2], s[2]), 'black')
                # plt.show()
                # print("\n")
        else:
            all_nodes.pop(0)
            
            
#         print("parent", parent)
#         print("final_seed", s)
#         print("\n")
#         print("cost", cost_dict)
        
# print("\n")
# print("final parent dict", parent_dict)
# print("\n")
# print("points", all_nodes)
# print("\n")
# print("visited_nodes", visited_nodes)
# print("\n")
# print("cost_dict", cost_dict)
# print("\n")
# print("Goal_Reached")
# print("dict", parent_dict)

path = []
path.append((s[0], s[1], s[2]))
# print("parent of goal", parent)
while parent != 'okay':
    temp = parent_dict.get(parent)
    path.append(parent)
#     print(path)
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
#     ax.plot3D((parent_list[l][0], seed_list[l][0]), (parent_list[l][1], seed_list[l][1]), (parent_list[l][2], seed_list[l][2]),  'black')
#     l = l + 1
#     plt.show()
#     plt.pause(0.000000000000000000000000000000000001)




ax.plot3D(x_path, y_path, z_path, "-r")
print(path)
ax.set_xlabel('x-axis') 
ax.set_ylabel('y-axis') 
ax.set_zlabel('z-axis') 
plt.show()
plt.pause(5)
plt.savefig("output.png")
plt.close()

