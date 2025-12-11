import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.serialization import deserialize_message
import open3d as o3d
import numpy as np
import os
import struct

# 参数配置
bag_path = "rosbag2_2025_12_10-17_05_21/rosbag2_2025_12_10-17_05_21_0.mcap"
topic_name = "/camera/camera/depth/color/points"
output_dir = "ply_output"
os.makedirs(output_dir, exist_ok=True)

# 动态导入 PointCloud2 类型
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

pkg, msg = topic_type.split('/')[0], topic_type.split('/')[-1]
exec(f"from {pkg}.msg import {msg}")
MsgType = eval(msg)

# 自动解析字段（RGB / packed RGB / grey）
def parse_pointcloud2(msg):
    field_names = [f.name for f in msg.fields]

    # Case 1: 直接含有 r g b 字段
    if set(["r", "g", "b"]).issubset(field_names):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
            x, y, z, r, g, b = p
            pts.append([x, y, z, r/255., g/255., b/255.])
        return np.array(pts)

    # Case 2: RealSense packed rgb float32 ("rgb")
    elif "rgb" in field_names:
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = p
            # unpack rgb float32
            packed = struct.unpack('I', struct.pack('f', rgb))[0]
            r = (packed >> 16) & 255
            g = (packed >> 8) & 255
            b = packed & 255
            pts.append([x, y, z, r/255., g/255., b/255.])
        return np.array(pts)

    # Case 3: 无 RGB，只含 XYZ
    elif set(["x","y","z"]).issubset(field_names):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            pts.append([x, y, z, 0.6, 0.6, 0.6])  # 灰色
        return np.array(pts)

    else:
        print("[ERROR] Unknown pointcloud format")
        return np.zeros((0, 6))


# 导出所有帧点云
count = 0

while reader.has_next():
    topic, data, stamp = reader.read_next()

    if topic != topic_name:
        continue

    msg = deserialize_message(data, MsgType)

    pts = parse_pointcloud2(msg)

    if pts.size == 0:
        print(f"[WARN] Empty frame {count}")
        continue

    # 保存为 PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:])

    filename = os.path.join(output_dir, f"cloud_{count:04d}.ply")
    o3d.io.write_point_cloud(filename, pcd)
    print(f"[SAVE] {filename}, points = {len(pts)}")

    count += 1

print(f"\n[DONE] Exported {count} frames → {output_dir}")
