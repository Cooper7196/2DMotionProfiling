import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def dist(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


curvatureAtStart = [0, 0]
curvatureAtEnd = [0, 0]


quinticHermite = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 1, 2, 3, 4, 5],
    [0, 0, 2, 6, 12, 20]
])
inversedQuinticHermite = np.linalg.inv(quinticHermite)

points = [Point(0, 0), Point(102.39793077501882, 97.96392024078256)]
headings = [(90, 97.96392024078256), (-90, 102.39793077501882)]

path = []
derivative = []
dDerivative = []
curvature = []

for i in range(len(points) - 1):
    controls = np.array([
        [points[i].x, points[i].y],
        [np.cos(np.radians(headings[i][0])) * headings[i][1], np.sin(np.radians(headings[i][0])) * headings[i][1]],
        curvatureAtStart,
        [points[i + 1].x, points[i + 1].y],
        [np.cos(np.radians(headings[i + 1][0])) * headings[i][1], np.sin(np.radians(headings[i + 1][0])) * headings[i][1]],
        curvatureAtEnd
    ])
    controls = np.array([
    [
        0,
        0
    ],
    [
        0,
        41.15316027088036
    ],
    [
        0,
        0
    ],
    [
        48.91267870579384,
        41.15316027088036
    ],
    [
        0,
        -41.15316027088036
    ],
    [
        0,
        0
    ]
])
    print(inversedQuinticHermite.tolist())
    coefficients = np.matmul(inversedQuinticHermite, controls)
    print(coefficients.tolist())
    for time in range(100):
        t = time / 100
        path.append(Point(
            np.matmul(coefficients[:, 0], np.array([1, t, t ** 2, t ** 3, t ** 4, t ** 5])),
            np.matmul(coefficients[:, 1], np.array([1, t, t ** 2, t ** 3, t ** 4, t ** 5]))
        ))
        derivative.append(Point(
            np.matmul(coefficients[:, 0], np.array([0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])),
            np.matmul(coefficients[:, 1], np.array([0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4]))
        ))
        dDerivative.append(Point(
            np.matmul(coefficients[:, 0], np.array([0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3])),
            np.matmul(coefficients[:, 1], np.array([0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3]))
        ))
        curvature.append((derivative[-1].x * dDerivative[-1].y - derivative[-1].y * dDerivative[-1].x) / (derivative[-1].x ** 2 + derivative[-1].y ** 2) ** 1.5)


fig, axs = plt.subplots(2, 1)

axs[0].plot([point.x for point in path], [point.y for point in path])

for i, point in enumerate(points):
    axs[0].plot(point.x, point.y, marker='o', color='r', ls='')
    axs[0].arrow(point.x, point.y, np.cos(np.radians(headings[i][0])) * headings[i][1], np.sin(np.radians(headings[i][0])) * headings[i][1], head_width=0.05, head_length=0.1, fc='k')
axs[0].set_title('Path')
axs[0].set_aspect('equal', 'box')

axs[1].plot(curvature)
axs[1].set_title('Curvature')

plt.show()

print(inversedQuinticHermite)
for i in range(len(inversedQuinticHermite)):
    print(str(list([round(i) for i in inversedQuinticHermite[i]])) + ",")