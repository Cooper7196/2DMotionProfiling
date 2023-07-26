import time
import matplotlib.pyplot as plt
import math
import json
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def dist(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class CubicBezier:
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
        
    def get_arc_length_at_t(self, t):
        # return self.lengths[int(t * self.len)]

        # Use linear interpolation to find the arc length at t
        return self.map(t) * self.length
    
    def get_t_at_arc_length(self, arc_length):
        # Use binary search to find the t value that corresponds to arc_length
        target = arc_length
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
    
    def get_length(self):
        return self.length
    def get_curvature(self, t):
        df = self.df(t)
        ddf = self.ddf(t)
        return (df[0] * ddf[1] - df[1] * ddf[0]) / (df[0]**2 + df[1]**2)**(3/2)


class QuinticBezier:
    def __init__(self, p0, p1, p2, p3, p4, p5, length=100):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.P = np.array(
            [
                [p0.x, p0.y],
                [p1.x, p1.y],
                [p2.x, p2.y],
                [p3.x, p3.y],
                [p4.x, p4.y]
                [p5.x, p5.y]
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

    def f(self, t):
        M = np.array(
                [[-1, 5, -10, 10, -5, 1,],
                [5, -20, 30, -20, 5, 0,],
                [-10, 30, -30, 10, 0, 0,],
                [10, -20, 10, 0, 0, 0,],
                [-5, 5, 0, 0, 0, 0,],
                [1, 0, 0, 0, 0, 0,]])
        T = np.array([t**5, t**4, t**3, t**2, t, 1])
        return np.matmul(np.matmul(T, M), self.P)
    
    def df(self, t):
        M = np.array(
                [[-5, 25, -50, 50, -25, 5,],
                [20, -80, 120, -80, 20, 0],
                [-30, 90, -90, 30, 0, 0],
                [20, -40, 20, 0, 0, 0],
                [-5, 5, 0, 0, 0, 0]])
        T = np.array([t**4, t**3, t**2, t, 1])
        return np.matmul(np.matmul(T, M), self.P)

    def ddf(self, t):
        M = np.array(
                [[-20, 100, -200, 200, -100, 20,],
                [60, -240, 360, -240, 60, 0],
                [-60, 180, -180, 60, 0, 0],
                [20, -40, 20, 0, 0, 0]])
        T = np.array([t**3, t**2, t, 1])
        return np.matmul(np.matmul(T, M), self.P)
class Spline:
    def __init__(self, curves):
        self.curves = curves
        self.lengths = [curve.get_length() for curve in curves]
        self.length = sum(self.lengths)
    
    def get_length(self):
        return self.length
    
    def get_curvature(self, t):
        curveIndex = int(t - 0.00000001)
        if curveIndex > len(self.curves) - 1:
            curveIndex = len(self.curves) - 1
        return self.curves[curveIndex].get_curvature(t - curveIndex)
    
    def get_t_at_arc_length(self, arc_length):
        for i, length in enumerate(self.lengths):
            if arc_length < sum(self.lengths[:(i + 1)]):
                return self.curves[i].get_t_at_arc_length(arc_length - sum(self.lengths[:i])) + i
        return len(self.curves)


class Constraints:
    def __init__(self, track_width, max_velocity, max_accel, max_decel=0, max_jerk=0):
        self.track_width = track_width
        self.max_velocity = max_velocity
        self.max_accel = max_accel
        self.max_decel = max_decel if max_decel != 0 else max_accel
        self.max_jerk = max_jerk
    
    def max_speed(self, curvature):
        return ((2 * self.max_velocity / self.track_width) * self.max_velocity) / (
            np.abs(curvature) * self.max_velocity + (2 * self.max_velocity / self.track_width)
        )
        return min(self.max_velocity, math.sqrt(self.max_accel / abs(curvature)))
    
    def wheel_speeds(self, velocity, angular_velocity):
        return (velocity - angular_velocity * (0.5 * self.track_width), velocity + angular_velocity * (0.5 * self.track_width))

class TrapezoidalProfile:
    def __init__(self, constraints, length, start_vel=0, end_vel=0):
        self.constraints = constraints
        self.length = length
        self.start_vel = start_vel
        self.end_vel = end_vel

        non_cruise_dist = constraints.max_velocity ** 2 / (2 * constraints.max_accel) + constraints.max_velocity ** 2 / (2 * constraints.max_decel)

        self.cruise_vel = constraints.max_velocity if (non_cruise_dist < length) else math.sqrt(2 * (length * constraints.max_accel * constraints.max_decel) / (constraints.max_accel + constraints.max_decel))

        self.accel_dist = (self.cruise_vel ** 2 - start_vel ** 2) / (2 * constraints.max_accel)
        self.decel_dist = length + (end_vel ** 2 - self.cruise_vel ** 2) / (2 * constraints.max_decel)
        
        # print("Profile Info: ", self.accel_dist, self.decel_dist, self.cruise_vel, self.length)
        
    def get_velocity_at_d(self, d):
        if d > self.length:
            d = self.length
        if d < self.accel_dist:
            return math.sqrt(self.start_vel **2 + 2 * self.constraints.max_accel * d)
        elif d < self.decel_dist:
            return self.cruise_vel
        else:
            return math.sqrt(self.cruise_vel **2 + 2 * -self.constraints.max_decel * (d - self.decel_dist))
    






with open("path.jerryio (1).txt", "r") as f:
    path = f.read().splitlines()

comments = [line for line in path if line[0] == '#']
print(comments[1])
# path = [line.split(',') for line in path if line[0] != '#']
# path = [(float(line[0]), float(line[1])) for line in path]

pathData = json.loads(comments[1].replace("#PATH.JERRYIO-DATA ", ''))

segments = pathData['paths'][0]['segments']
beziers = []
for segment in segments:
    points = []
    for control in segment['controls']:
        print(control['x'], control['y'])
        points.append(Point(control['x'], control['y']))
    beziers.append(CubicBezier(points[0], points[1], points[2], points[3], 100))
path = Spline(beziers)
# Motion Profiling
angular_accel_arr = []
def generate(constraints : Constraints, path, dt=0.001):
    trajectory = []
    length = path.get_length()
    profile = TrapezoidalProfile(constraints, length)

    t = 0
    dist = 0.01

    vel = 0.00
    last_angular_vel = 0.00
    angular_vel = 0.00
    angular_accel = 0.00
    decel_dist = 0
    tmp_dist = 0
    i = 0
    while dist <= length:
        t = path.get_t_at_arc_length(dist)
        if i % 100 == 0:
            print(i, t, dist, tmp_dist, length)
        curvature = path.get_curvature(t)

        angular_vel = vel * curvature
        angular_accel = (angular_vel - last_angular_vel) / dt
        last_angular_vel = angular_vel

        max_accel = constraints.max_accel - abs(angular_accel * constraints.track_width / 2)
        max_decel = constraints.max_decel - abs(angular_accel * constraints.track_width / 2)

        # calc dist to decelerate
        if tmp_dist < length:
            tmp_dist = dist
            tmp_vel = vel
            decel_dist = 0
            tmp_angular_accel = 0
            tmp_last_angular_vel = angular_vel
            max_decel_arr = []
            while tmp_dist <= length and tmp_vel > 0 :
                tmp_t = path.get_t_at_arc_length(tmp_dist)
                tmp_curvature = path.get_curvature(tmp_t)

                tmp_angular_vel = tmp_vel * tmp_curvature
                tmp_angular_accel = (tmp_angular_vel - tmp_last_angular_vel) / dt
                tmp_last_angular_vel = tmp_angular_vel

                tmp_max_decel = constraints.max_decel - abs(tmp_angular_accel * constraints.track_width / 2)

                tmp_vel = min((constraints.max_speed(tmp_curvature), tmp_vel - tmp_max_decel * dt))

                tmp_dist += tmp_vel * dt
                decel_dist += tmp_vel * dt
                
                max_decel_arr.append(tmp_max_decel)
            if tmp_dist < length:
                tmp_dist = 0
        
        vel = min((constraints.max_speed(curvature), vel + max_accel * dt, (vel - max_decel * dt if tmp_dist >= length else float('inf'))))
        # if tmp_dist >= dist and len(max_decel_arr) > 1:
        #     plt.clf()
        #     plt.plot(max_decel_arr, '.')
        #     plt.show()
        # print(angular_accel)
        angular_accel_arr.append(angular_accel * constraints.track_width / 2)
        delta_d = vel * dt
        dist += delta_d

        trajectory.append((vel, angular_vel, curvature))
        i += 1
    trajectory.append((0, 0, 0))
    
    
    # forward pass
    # for i in range(len(trajectory) - 1):
    #     angular_accel = (trajectory[i + 1][1] - trajectory[i][1]) / dt
    #     max_accel = constraints.max_accel - abs(angular_accel * constraints.track_width / 2)
    #     max_decel = constraints.max_decel - abs(angular_accel * constraints.track_width / 2)
    #     # max_accel = constraints.max_accel
    #     if (trajectory[i + 1][0] - trajectory[i][0]) / dt > max_accel:
    #         print("test")
    #         trajectory[i + 1] = (min(constraints.max_speed(trajectory[i][2]), (trajectory[i][0] + max_accel * dt)), trajectory[i + 1][1], trajectory[i + 1][2])
    #     if (trajectory[i + 1][0] - trajectory[i][0]) / dt < -max_decel:
    #         print("test1", i, (trajectory[i + 1][0] - trajectory[i][0]))
    #         trajectory[i + 1] = (min((trajectory[i][0] - max_decel * dt, )), trajectory[i + 1][1], trajectory[i + 1][2])
    #     curLeft, curRight = constraints.wheel_speeds(trajectory[i][0], trajectory[i][1])
    #     nextLeft, nextRight = constraints.wheel_speeds(trajectory[i + 1][0], trajectory[i + 1][1])
    # trajectory.append((0, 0, 0))
    # # backward pass
    # for i in range(len(trajectory) - 1, 0, -1):
    #     angular_accel = (trajectory[i - 1][1] - trajectory[i][1]) / dt
    #     max_accel = constraints.max_accel - abs(angular_accel * constraints.track_width / 2)
    #     max_decel = constraints.max_decel - abs(angular_accel * constraints.track_width / 2)
    #     if i % 10 == 0:
    #         print(i, max_accel, max_decel)
    #     # max_accel = constraints.max_accel
    #     if (trajectory[i - 1][0] - trajectory[i][0]) / dt > max_accel:
    #         print("test2", i, (trajectory[i - 1][0] - trajectory[i][0]) / dt)
    #         trajectory[i - 1] = (min(constraints.max_speed(trajectory[i][2]), (trajectory[i][0] + max_accel * dt)), trajectory[i - 1][1], trajectory[i - 1][2])
    #     if (trajectory[i - 1][0] - trajectory[i][0]) / dt < -max_decel:
    #         print("test3", i, (trajectory[i - 1][0] - trajectory[i][0]) / dt)
    #         trajectory[i - 1] = (min((trajectory[i][0] - max_decel * dt, )), trajectory[i - 1][1], trajectory[i - 1][2])

    return [trajectory[i] for i in range(len(trajectory)) if i % (0.01 / dt) == 0]


startTime = time.time()
constraints = Constraints(15, 62.8318, 100)




trajectory = generate(constraints, path)
print(time.time() - startTime)
accel = []
for i in range(len(trajectory) - 1):
    accel.append((trajectory[i + 1][0] - trajectory[i][0]) / 0.01)
vel = []
d = 0
# print("test", bezier.get_curvature(0.75))
angular_accel_arr.remove(max(angular_accel_arr))
angular_accel_arr.remove(min(angular_accel_arr))
# angular_accel_arr.remove(min(angular_accel_arr))

accel.remove(min(accel))
accel.remove(min(accel))
accel.remove(min(accel))
accel.remove(min(accel))
accel.remove(min(accel))
accel.remove(max(accel))

angular_accel_arr = [angular_accel_arr[i] for i in range(len(angular_accel_arr)) if i % (0.01 / 0.001) == 0]

plt.clf()
plt.plot([i[0] for i in trajectory],  '.')
# plt.plot(angular_accel_arr, 'x')
plt.plot(accel, 'o')
plt.show()

# plt.plot(angular_accel_arr, '.')
# plt.show()
# calulate acceleration
accel = []
# for i in range(1, len(trajectory)):
#     trajectory[i] = (0, trajectory[i][1])
for i in range(1, len(trajectory)):
    left, right = constraints.wheel_speeds(trajectory[i][0], trajectory[i][1])
    lastLeft, lastRight = constraints.wheel_speeds(trajectory[i - 1][0], trajectory[i - 1][1])
    accel.append(((left - lastLeft) / 0.01, (right - lastRight) / 0.01))


accel.remove(min(accel, key=lambda x: x[0]))
accel.remove(min(accel, key=lambda x: x[1]))

# for i in range(len(accel)):
#     print(accel[i][1] / angular_accel_arr[i])

plt.clf()
plt.plot([i[0] for i in accel], '.')
plt.plot([i[1] for i in accel], 'x')
plt.show()



print(f"Total Time: {len(trajectory) * 0.01} Total Distance: {path.get_length()}")
with open("profile.txt", "w") as file:
    for i in range(len(trajectory)):
        left, right = constraints.wheel_speeds(trajectory[i][0], trajectory[i][1])
        file.write(f"0,0,{left},{right}\n")