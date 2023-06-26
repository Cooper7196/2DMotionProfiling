import matplotlib.pyplot as plt
import json
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def dist(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class Bezier:
    def __init__(self, p0, p1, p2, p3, length=100):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.P = np.array(
            [
                [p0.x, p0.y],
                [p1.x, p1.y],
                [p2.x, p2.y],
                [p3.x, p3.y]
            ])
        self.len = length
        self.lengths = [0 for i in range(self.len + 1)]

        ox, oy = self.f(0)
        clen = 0
        for i in range(self.len + 1):
            x, y = self.f(i / float(self.len))
            dx = ox - x
            dy = oy - y
            clen += (dx**2 + dy**2)**0.5
            self.lengths[i] = clen
            ox = x
            oy = y
        self.length = clen

    def get_uniform_path(self):
        path = []
        for i in range(self.len + 1):
            x, y = self.mf(i / float(self.len))
            path.append(Point(x, y))
        return path
    
    def get_uniform_path_derivative(self):
        path = []
        for i in range(self.len + 1):
            x, y = self.mdf(i / float(self.len))
            path.append(Point(x, y))
        return path
    
    def get_uniform_path_2nd_derivative(self):
        path = []
        for i in range(self.len + 1):
            x, y = self.mddf(i / float(self.len))
            path.append(Point(x, y))
        return path
    
    def map(self, u):
        target = u * self.lengths[self.len]
        low = 0
        high = self.len
        index = 0
        while low < high:
            index = int(low + (high - low) // 2)
            if self.lengths[index] < target:
                low = index + 1
            else:
                high = index
        if self.lengths[index] > target:
            index -= 1
        lengthBefore = self.lengths[index]
        if lengthBefore == target:
            return index / float(self.len)
        else:
            return (index + (target - lengthBefore) / (self.lengths[index + 1] - lengthBefore)) / float(self.len)
    
    def mf(self, u):
        return self.f(self.map(u))
    def mdf(self, u):
        return self.df(self.map(u))
    def mddf(self, u):
        return self.ddf(self.map(u))


    def f(self, t):
        M = np.array(
                [[-1, 3, -3, 1,],
                [3, -6, 3, 0,],
                [-3, 3, 0, 0,],
                [1, 0, 0, 0,]])
        T = np.array([t**3, t**2, t, 1])
        return np.matmul(np.matmul(T, M), self.P)
    
    def df(self, t):
        M = np.array(
                [[-3, 9, -9, 3,],
                [6, -12, 6, 0,],
                [-3, 3, 0, 0,]])
        T = np.array([t**2, t, 1])
        return np.matmul(np.matmul(T, M), self.P)
    
    def ddf(self, t):
        M = np.array(
                [[-6, 18, -18, 6,],
                [6, -12, 6, 0,]])
        T = np.array([t, 1])
        return np.matmul(np.matmul(T, M), self.P)
    
with open("path.jerryio (1).txt", "r") as f:
    path = f.read().splitlines()

comments = [line for line in path if line[0] == '#']
print(comments[1])
# path = [line.split(',') for line in path if line[0] != '#']
# path = [(float(line[0]), float(line[1])) for line in path]

pathData = json.loads(comments[1].replace("#PATH.JERRYIO-DATA ", ''))

segments = pathData['paths'][0]['segments']
print(segments)
# print(json.dumps(segments, indent=4, sort_keys=True))
path = []
for segment in segments:
    points = []
    for control in segment['controls']:
        print(control['x'], control['y'])
        points.append(Point(control['x'], control['y']))
    bezier = Bezier(points[0], points[1], points[2], points[3], 100)
    path += bezier.get_uniform_path()
path = []
bezier = Bezier(Point(0, 0), Point(0, 1), Point(1, 0), Point(1, 1), 100)
path += bezier.get_uniform_path()

# derivaive of the path
dpath = bezier.get_uniform_path_derivative()
# 2nd derivative of the pathe
ddpath = bezier.get_uniform_path_2nd_derivative()
# Curvature of the path
curvature = [Point(i, (1/(dpath[i].x*ddpath[i].y - dpath[i].y*ddpath[i].x)) / (dpath[i].x**2 + dpath[i].y**2)**(3/2)) for i in range(len(dpath) - 1)]
# angularVel = 
curvature.remove(max(curvature, key=lambda p: p.y))
curvature.remove(min(curvature, key=lambda p: p.y))

# Draw the path
fig, ax = plt.subplots(2, 2)
ax[0][0].plot([p.x for p in path], [p.y for p in path], '.')
ax[0][0].plot(path[0].x, path[0].y, 'rp', markersize=14)
# ax[0][0].set_aspect('equal')
ax[0][0].set_title('Path')

# ax2 = ax[0][0].twinx()  # instantiate a second axes that shares the same x-axis

ax[1][0].plot([p.x for p in curvature[:100]], [p.y for p in curvature[:100]], '.')
ax[1][0].set_title('Curvature of the path')

ax[0][1].plot([p.x for p in dpath[:100]], [p.y for p in dpath[:100]], '.')
ax[0][1].set_aspect('equal')
ax[0][1].set_title('Derivative of the path')

ax[1][1].plot([p.x for p in ddpath[:100]], [p.y for p in ddpath[:100]], '.')
ax[1][1].set_aspect('equal')
ax[1][1].set_title('2nd Derivative of the path')
print(min([p.y for p in curvature]), max([p.y for p in curvature]))
plt.show()

# fig, ax = plt.subplots(2)

# ax[0].plot([p.x for p in dpath], [p.y for p in dpath], '.')
# ax[0].set_title('Derivative of the path')

# ax[1].plot([p.x for p in dpathTest], [p.y for p in dpathTest], '.')
# ax[1].set_title('2nd Derivative of the path')
# plt.show()
