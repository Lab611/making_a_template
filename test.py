import numpy as np

# 定义点的坐标
P0 = np.array([-44.388, 4.721, -113.733])
P1 = np.array([4.077, -105.777, -14.529])
P2 = np.array([23.364, 41.891, -106.438])

# P0 = np.array([ 6.47392167e+01, -6.92829891e+01,  5.56123917e-04])
# P1 = np.array([0, 0, 0])
# P2 = np.array([175, 0, 0])

# 计算向量P1P0和P1P2
P1P0 = P0 - P1
P1P2 = P2 - P1

# 计算P1P2的单位向量
P1P2_unit = P1P2 / np.linalg.norm(P1P2)

# 计算点P0到直线P1P2的距离
distance = np.linalg.norm(np.cross(P1P2, P1P0)) / np.linalg.norm(P1P2)

print("Distance from point P0 to the line formed by points P1 and P2:", distance)