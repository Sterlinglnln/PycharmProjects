import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.serialization import deserialize_message
import open3d as o3d
import numpy as np
import os
import struct

# -------------------------------
# 修改这里
# -------------------------------
bag_path = "rosbag2_2025_12_08-20_12_12"
topic_name = "/camera/camera/depth/color/points"
output_ply = "single_frame_preprocessed.ply"


# -------------------------------
# Step 1: 打开 MCAP
# -------------------------------
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

topics = reader.get_all_topics_and_types()
topic_type = None

for t in topics:
    if t.name == topic_name:
        topic_type = t.type
        print(f"[INFO] Topic = {topic_name}, type = {topic_type}")
        break

if topic_type is None:
    raise RuntimeError(f"未找到话题 {topic_name}")

pkg, msg = topic_type.split('/')[0], topic_type.split('/')[-1]
exec(f"from {pkg}.msg import {msg}")
MsgType = eval(msg)


# -------------------------------
# Step 2: PointCloud2 字段自动解析
# -------------------------------
def parse_pointcloud2(msg):
    field_names = [f.name for f in msg.fields]

    # Case 1: 显式 RGB 字段
    if set(["r", "g", "b"]).issubset(field_names):
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z","r","g","b"), skip_nans=True):
            x, y, z, r, g, b = p
            pts.append([x, y, z, r/255., g/255., b/255.])
        return np.array(pts)

    # Case 2: RealSense packed RGB (float)
    if "rgb" in field_names:
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z","rgb"), skip_nans=True):
            x, y, z, rgb = p
            packed = struct.unpack('I', struct.pack('f', rgb))[0]
            r = (packed >> 16) & 255
            g = (packed >> 8) & 255
            b = packed & 255
            pts.append([x, y, z, r/255., g/255., b/255.])
        return np.array(pts)

    # Case 3: 只有 XYZ
    if set(["x","y","z"]).issubset(field_names):
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
            x, y, z = p
            pts.append([x, y, z, 0.6, 0.6, 0.6])
        return np.array(pts)

    print("[ERROR] Unsupported PointCloud2 format")
    return np.zeros((0, 6))


# -------------------------------
# Step 3: 读取第一帧
# -------------------------------
first_frame_pts = None

while reader.has_next():
    topic, data, stamp = reader.read_next()
    if topic != topic_name:
        continue

    msg = deserialize_message(data, MsgType)
    pts = parse_pointcloud2(msg)

    if pts.size == 0:
        continue

    first_frame_pts = pts
    print(f"[INFO] 读取第一帧点云: {pts.shape[0]} points")
    break

if first_frame_pts is None:
    raise RuntimeError("未读取到点云数据！")


# -------------------------------
# Step 4: 点云预处理（距离裁剪 + 去噪）
# -------------------------------
print("[INFO] 开始点云预处理 ...")

# 转 open3d 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(first_frame_pts[:, :3])
pcd.colors = o3d.utility.Vector3dVector(first_frame_pts[:, 3:])

# 1) 距离裁剪（0.2m - 1.2m）
pts = np.asarray(pcd.points)
dist = np.linalg.norm(pts, axis=1)
mask = (dist > 0.2) & (dist < 1.2)
pcd = pcd.select_by_index(np.where(mask)[0])
print(f"[INFO] 距离裁剪后: {len(pcd.points)} points")

# # 2) 半径滤波
# pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=0.03)
# print(f"[INFO] 半径滤波后: {len(pcd.points)} points")
#
# # 3) 统计滤波
# pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
# print(f"[INFO] 统计滤波后: {len(pcd.points)} points")


# -------------------------------
# Step 5: 保存结果
# -------------------------------
o3d.io.write_point_cloud(output_ply, pcd)
print(f"[SAVE] 已保存 → {output_ply}")
print("[DONE] 提取第一帧 + 点云预处理完成")
