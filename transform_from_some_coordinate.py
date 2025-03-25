import numpy as np
import cv2
import numpy as np
from scipy.linalg import svd
import json

# 启用测试输出
TEST_OUTPUT = True



#     计算两个三维点集之间的仿射变换。（需要存储新旧球的顺序，重新排序）  ??????好像有问题？？
def find_3d_affine_transform(in_points_Reorder, out_points):
    """
    计算两个三维点集之间的仿射变换。

    参数:
    in_points (numpy.ndarray): 输入的三维点集，形状为 (3, N)。
    out_points (numpy.ndarray): 输出的三维点集，形状为 (3, N)。

    返回:
    numpy.ndarray: 仿射变换矩阵，形状为 (4, 4)。
    """

    # 检查输入的两个矩阵的列数是否相同
    if in_points_Reorder.shape[1] != out_points.shape[1]:
        raise ValueError("Find3DAffineTransform(): input data mis-match")

    # 计算输入和输出点集之间的比例因子
    # dist_in = np.sum(np.linalg.norm(in_points[:, 1:] - in_points[:, :-1], axis=0))
    # dist_out = np.sum(np.linalg.norm(out_points[:, 1:] - out_points[:, :-1], axis=0))
    # if dist_in <= 0 or dist_out <= 0:
    #     return np.eye(4)
    #
    # scale = dist_out / dist_in
    # out_points /= scale

    # 计算输入和输出点集的中心点
    in_ctr = np.mean(in_points_Reorder.T, axis=1)
    out_ctr = np.mean(out_points.T, axis=1)

    print("提供的坐标的中心点：\n", in_ctr)
    print("模板坐标的中心点：\n", out_ctr)
    # 将点集平移到原点
    in_points_centered = in_points_Reorder.T - in_ctr.reshape(3, 1)
    out_points_centered = out_points.T - out_ctr.reshape(3, 1)

    # 计算协方差矩阵并进行SVD分解
    cov_matrix = in_points_centered @ out_points_centered.T
    U, s, Vt = svd(cov_matrix)

    # 计算旋转矩阵
    d = np.linalg.det(Vt @ U.T)
    I = np.eye(3)
    I[2, 2] = d
    R = Vt @ I @ U.T

    # 计算最终的仿射变换矩阵
    # T = scale * (out_ctr - R @ in_ctr)
    # transform_matrix = np.eye(4)
    # transform_matrix[:3, :3] = scale * R
    # transform_matrix[:3, 3] = T

    # 计算平移向量（注意这里没有使用缩放因子）
    T = out_ctr - R @ in_ctr

    # 构建最终的仿射变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T

    return transform_matrix
#     计算点到直线的距离
def point_to_line_distance_3D(point, line_point1, line_point2):
    # 计算点到直线的距离
    direction_vector = line_point2 - line_point1
    distance = np.linalg.norm(np.cross(direction_vector, point - line_point1)) / np.linalg.norm(direction_vector)
    return distance
#     计算距离矩阵
################### 问题：这里计算的模板坐标系以外还需要计算一个原始顺序对应的模板坐标系，才能计算仿射变换
def get_distance_matrix(p3D_list):
    row_num = len(p3D_list)
    dis = np.zeros((row_num, row_num))

    for i in range(row_num):
        dis[i][i] = 0.0
        for j in range(i + 1, row_num):
            A = np.array(p3D_list[i])
            B = np.array(p3D_list[j])
            distance = cv2.norm(A - B)  # 使用OpenCV的norm函数计算两点之间的距离

            # 测试输出
            # print(f"第{i + 1}个小球 {A} 与第{j + 1}个小球 {B} 的距离是 {distance}")

            dis[i][j] = distance
            dis[j][i] = distance

    # 测试输出
    # print("距离矩阵为:")
    # np.set_printoptions(formatter={'float': '{:0.2f}'.format})
    # print(dis)

    return dis
#     计算模板坐标系
def get_template_3Ds(p3D_list_points,design_poi3Ds ):
    Zero = 0.0
    row_num = len(p3D_list_points) # row_num=点个数=行数

    # S1：找到前三个点
    # 计算中心点center_point
    center_point = np.mean(p3D_list_points, axis=0)
    print("center_point:\n", center_point)
    # 找到距离中心点最远的点，作为P0
    max_distance2center_poin = 0
    row_P0 = 0
    for i, point in enumerate(p3D_list_points):
        distance = np.linalg.norm(point - center_point)
        if distance > max_distance2center_poin:
            max_distance2center_poin = distance
            row_P0 = i
    print("row_P0:\n", row_P0)
    # 创建新的点集，并添加P0
    p3Ds_temple = [np.array([Zero, Zero, Zero])]
    in_points_Reorder = [p3D_list_points[row_P0]] # 重新排序的原始点集 ！！！！新加
    print("已添加P0的p3Ds_temple:")
    print(p3Ds_temple)
 # ----------------------------- 第一个点创建完毕 ---------------------
    # 调用函数并获取距离矩阵
    d_matrix = get_distance_matrix(p3D_list_points)

    # 找到距离P0最远的点，作为P1
    max_distance2P0 = 0
    row_P1 = 0

    for j in range(row_num):
        if d_matrix[j][row_P0] > max_distance2P0:
            max_distance2P0 = d_matrix[j][row_P0]
            row_P1 = j
    print("row_P1:\n", row_P1)
    # 添加P1
    a = max_distance2P0
    p3Ds_temple.append(np.array([a, Zero, Zero]))
    print("已添加P1的p3Ds_temple:")
    print(p3Ds_temple)
    print("max_distance2P0:")
    print("a：", a)
    in_points_Reorder.append(p3D_list_points[row_P1]) # 重新排序的原始点集 ！！！！新加

    # P0和P1的坐标
    P0 = p3D_list_points[row_P0]
    P1 = p3D_list_points[row_P1]

    # 计算P0P1的方向向量
    direction_Vec_P1P0 = P1 - P0

    # 找到距离直线P0P1最远的点，作为P2
    max_distance2line_P0P1 = 0
    row_P2 = 0
    for k, point in enumerate(p3D_list_points):
        distance = point_to_line_distance_3D(point, P0, P1)
        if distance > max_distance2line_P0P1:
            max_distance2line_P0P1 = distance
            row_P2 = k
    print("row_P2:\n", row_P2)
    # P2的坐标
    P2 = p3D_list_points[row_P2]
    print(f"\nP0: {P0}")
    print(f"P1: {P1}")
    print(f"P2: {P2}")
    in_points_Reorder.append(p3D_list_points[row_P2]) # 重新排序的原始点集 ！！！！新加
    # 计算平面方程AX+BY+CZ+D=0的系数A, B, C和D
    A, B, C = direction_Vec_P1P0
    D = -np.dot(direction_Vec_P1P0, P2)

    # 计算P0到平面的距离（即b）
    b = np.abs(A * P0[0] + B * P0[1] + C * P0[2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
    print("b:",b)
    # c就是max_distance2line_P0P1
    c = max_distance2line_P0P1

    # 添加P2
    p3Ds_temple.append(np.array([b, c, Zero]))
    print("已添加P0、P1、P2的distance:")
    print(get_distance_matrix(p3Ds_temple))
    # S2：找到三个原点平面

    """
    // XOZ平面：
    // 计算过点P2的平面AX + BY + CZ + D = 0
    和直线P0P1的交点P2h
        假设直线
        P0P1
        的两个端点是
        P0(x0, y0, z0), P1(x1, y1, z1)，那么直线的参数方程可以表示为：
        x = x0 + t(x1 - x0)
        y = y0 + t(y1 - y0)
        z = z0 + t(z1 - z0)


        代入AX + BY + CZ + D = 0，得：
        t = -(A * P0.x + B * P0.y + C * P0.z + D) / (A * (P1.x - P0.x) + B * (P1.y - P0.y) + C * (P1.z - P0.z))
    """
    # 计算参数t
    numerator = -(A * P0[0] + B * P0[1] + C * P0[2] + D)
    denominator = (A * (P1[0] - P0[0]) + B * (P1[1] - P0[1]) + C * (P1[2] - P0[2]))
    t = numerator / denominator

    # 计算交点P2h的坐标
    P2h =( P0 + t * (P1 - P0))

    # 输出交点P2h
    print(" 原始坐标中P2h 的坐标:", P2h)
    print(" 原始坐标中P0 的坐标:", P0)
    print(" 新坐标系中P2h到P0:",b)
    print(" 旧坐标系中P2h到P0:",np.linalg.norm(P2h - P0))

    # 计算 P2hP2
    direction_Vec_P2P2h = P2 - P2h
    A_XOZ, B_XOZ, C_XOZ = direction_Vec_P2P2h
    D_XOZ = -np.dot(direction_Vec_P2P2h, P0)
    print("(direction_Vec_P2P2h, P1:",direction_Vec_P2P2h, P1)
    # XOY平面：
    # 上述已知P0P1和P0P2向量
    # 计算叉积得到XOY平面的法向量
    direction_Vec_P2P0 = P2 - P0
    # 计算叉乘
    normalVec_XOY = np.cross(direction_Vec_P1P0, direction_Vec_P2P0)
    A_XOY, B_XOY, C_XOY = normalVec_XOY
    # 点乘，代入P0点求解D
    D_XOY = -np.dot(normalVec_XOY, P0)

    # YOZ平面：过p0垂直直线p0p1的平面
    # 计算平面方程AX + BY + CZ + D = 0的系数A, B, C和D
    A_YOZ, B_YOZ, C_YOZ = direction_Vec_P1P0

    D_YOZ = -np.dot(direction_Vec_P1P0, P0)


    # S3：计算其余点的模板坐标
    balls_num = 3
    excluded_rows = {row_P0, row_P1, row_P2}
    row_Flag = 0

    while balls_num < row_num:
        while row_Flag in excluded_rows:
            row_Flag += 1
            if row_Flag >= row_num:
                break

        if row_Flag < row_num:
            current_point = p3D_list_points[row_Flag]
            print("YOZ平面:\n", A_YOZ, B_YOZ, C_YOZ, D_YOZ)
            print("XOZ平面:\n", A_XOZ, B_XOZ, C_XOZ, D_XOZ)
            print("XOY平面:\n", A_XOY, B_XOY, C_XOY, D_XOY)
            # 计算当前点到各个平面的距离（需要具体的平面方程）
            distance_to_X = (A_YOZ * current_point[0] + B_YOZ * current_point[1] + C_YOZ * current_point[2] + D_YOZ) / np.sqrt(A_YOZ ** 2 + B_YOZ ** 2 + C_YOZ ** 2)
            distance_to_Y = (A_XOZ * current_point[0] + B_XOZ * current_point[1] + C_XOZ * current_point[2] + D_XOZ) / np.sqrt(A_XOZ ** 2 + B_XOZ ** 2 + C_XOZ ** 2)
            distance_to_Z = (A_XOY * current_point[0] + B_XOY * current_point[1] + C_XOY * current_point[2] + D_XOY) / np.sqrt(A_XOY ** 2 + B_XOY ** 2 + C_XOY ** 2)
            print(f"原坐标 {current_point}")
            print(f"距离 X: {distance_to_X}, Y: {distance_to_Y}, Z: {distance_to_Z}")
            # 更新当前点的坐标
            current_point = np.array([distance_to_X, distance_to_Y, distance_to_Z])
            print(f"新坐标 {current_point}")

            # 将新坐标添加到模板点集中
            p3Ds_temple.append(current_point)
            in_points_Reorder.append(p3D_list_points[row_Flag])  # 重新排序的原始点集 ！！！！新加
            # 更新balls_num和排除当前行
            balls_num += 1
            excluded_rows.add(row_Flag)

    # 使用np.vstack将列表转换为二维数组
    in_points_Array = np.vstack(in_points_Reorder)

    new_poi3Ds = []
    # 计算poi
    for p in design_poi3Ds:
        # 计算当前点到各个平面的距离（需要具体的平面方程）
        distance_to_X = (A_YOZ * p[0] + B_YOZ * p[1] + C_YOZ * p[
            2] + D_YOZ) / np.sqrt(A_YOZ ** 2 + B_YOZ ** 2 + C_YOZ ** 2)
        distance_to_Y = (A_XOZ * p[0] + B_XOZ * p[1] + C_XOZ * p[
            2] + D_XOZ) / np.sqrt(A_XOZ ** 2 + B_XOZ ** 2 + C_XOZ ** 2)
        distance_to_Z = (A_XOY * p[0] + B_XOY * p[1] + C_XOY * p[
            2] + D_XOY) / np.sqrt(A_XOY ** 2 + B_XOY ** 2 + C_XOY ** 2)
        print(f"原坐标 {p}")
        print(f"距离 X: {distance_to_X}, Y: {distance_to_Y}, Z: {distance_to_Z}")
        # 更新当前点的坐标
        p = np.array([distance_to_X, distance_to_Y, distance_to_Z])
        print(f"新坐标 {p}")

        # 将新坐标添加到模板点集中
        new_poi3Ds.append(p)



    return np.array(p3Ds_temple), in_points_Array, new_poi3Ds

def apply_transformation(current_Rt, design_poi3Ds):
    # 将变换矩阵应用于点集
    # 假设design_poi3Ds是Nx3的矩阵，需要扩展为Nx4（添加一列1）以应用4x4变换矩阵
    n = len(design_poi3Ds)
    design_poi3Ds_homogeneous = np.hstack((design_poi3Ds, np.ones((n, 1))))
    template_poi3Ds_homogeneous = current_Rt @ design_poi3Ds_homogeneous.T
    template_poi3Ds = template_poi3Ds_homogeneous.T[:, :3]
    return template_poi3Ds
def process_json(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    targets = data['Target']
    processed_targets = {}
        # 　S1 从json中读出design_ball3Ds、design_poi3Ds
    for target_name, target_data in targets.items():
        design_ball3Ds = np.array(target_data['points_ball'])
        design_poi3Ds = np.array(target_data['points_poi'])
        # 　S2 使用get_distance_matrix(design_ball3Ds)得到新的模板坐标系下的点 template_ball3Ds
        template_ball3Ds, in_points_Reorder, template_poi3Ds= get_template_3Ds(design_ball3Ds,design_poi3Ds)
        # 变换成二维数组，形状为(1, 3)
        template_poi3Ds = np.array(template_poi3Ds)
        print("in_points_Reorder:\n  ", in_points_Reorder)
        print("template_ball3Ds:\n  ", template_ball3Ds)
        # 　S3 得到Rt current_Rt
        current_Rt = find_3d_affine_transform(in_points_Reorder, template_ball3Ds)
        # 　S4 计算模板坐标系下兴趣点的坐标 template_poi3Ds
        template_poi3Ds_test = apply_transformation(current_Rt, design_poi3Ds)
        print("测试变换前:\n", design_ball3Ds)
        print("测试变换后:\n", apply_transformation(current_Rt, design_ball3Ds))
        print("模板制作直接计算的poi：\n",template_poi3Ds)
        print("应用仿射变换计算的poi：\n",template_poi3Ds_test)

        print("旧的 design_ball3Ds:\n", design_ball3Ds)
        print("新的 template_ball3Ds:\n", template_ball3Ds)
        print("旧的 design_ball3Ds_distance:\n", get_distance_matrix(design_ball3Ds))
        print("新的 template_ball3Ds_distance:\n", get_distance_matrix(template_ball3Ds))

        processed_target_data = {
            'points_ball': template_ball3Ds.tolist(),
            'points_poi': template_poi3Ds.tolist()
        }

        processed_targets[target_name] = processed_target_data

    output_data = {
        'Target': processed_targets
    }
    # 　S5 输出回json
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)






# 使用示例
input_json_path = 'Data/hqd手柄.json'
output_json_path = 'Data/output_hqd手柄.json'
process_json(input_json_path, output_json_path)

