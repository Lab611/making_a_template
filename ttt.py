import numpy as np

# 计算一个点到其他点的距离


# 新的poi
A0 = [0.0, 0.0, 0.0]
A1 = [175.00025963980738, 0.0, 0.0]
A2 = [54.99996042183354, 77.94202792843375, 0.0]
A3 = [139.9998759740524,-69.28298905171943,0.000556123916901445]
A4 = [81.17655785445984,13.282933606320098,-68.19413683200288] # 新的poi

# 旧的poi
A01 = [4.077, -105.777, -14.529]
A11 = [23.364, 41.891, -106.438]
A21 = [82.019, -50.777, -14.529]
A31 = [-44.388, 4.721, -113.733]
A41 = [0.0, 0.0, 0.0] # 旧的poi

# P2h11 = [54.99996042183354, 0,  0]
# P2h = [10.13861522, -59.3671594,  -43.41462207]
# P0 = [4.077, -105.777, -14.529]
# 计算两点之间的差值向量
difference0 = np.array(A4) - np.array(A0)
difference1 = np.array(A4) - np.array(A1)
difference2 = np.array(A4) - np.array(A2)
difference3 = np.array(A4) - np.array(A3)
difference01 = np.array(A41) - np.array(A01)
difference11 = np.array(A41) - np.array(A11)
difference21 = np.array(A41) - np.array(A21)
difference31 = np.array(A41) - np.array(A31)
# difference =  np.array(P2h) - np.array(P0)
# difference11 =  np.array(P2h11) - np.array(A1)
# 计算差值向量的模长，即两点之间的距离
distance0 = np.linalg.norm(difference0)
distance1 = np.linalg.norm(difference1)
distance2 = np.linalg.norm(difference2)
distance3 = np.linalg.norm(difference3)
distance01 = np.linalg.norm(difference01)
distance11 = np.linalg.norm(difference11)
distance21 = np.linalg.norm(difference21)
distance31 = np.linalg.norm(difference31)
# distance = np.linalg.norm(difference)
# distance11 = np.linalg.norm(difference11)
print(distance0,"，",distance1,"，", distance2,"，",  distance3)
print(distance01,"，",distance11,"，", distance21,"，",  distance31)
# print(distance)
# print(distance11)

#
# # A1 = [4.077, -105.777, -14.529]
# # A2 = [23.364, 41.891, -106.438]
# # A3 = [82.019, -50.777, -14.529]
#
# A1 = [0.0,0.0,0.0]
# A2 = [175.00025963980738,0.0,0.0]
# A3 = [54.99996042183354,77.94202792843375,0.0]
#
# # 计算两点之间的差值向量
# difference1 = np.array(A3) - np.array(A1)
# difference2 = np.array(A3) - np.array(A2)
#
# # 计算差值向量的模长，即两点之间的距离
# distance1 = np.linalg.norm(difference1)
# distance2 = np.linalg.norm(difference2)
#
# print(distance1,"\n", distance2,"\n")