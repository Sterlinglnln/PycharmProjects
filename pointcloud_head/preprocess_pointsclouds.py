# preprocess_pointclouds.py
import open3d as o3d
import os
import glob

# 原始点云目录（你前面导出的）
INPUT_DIR = "ply_output"
# 预处理后点云目录
OUTPUT_DIR = "pcd_down"

# 体素大小（单位：米，视你头部距离 0.5~0.7m，3mm～5mm 比较合适）
VOXEL_SIZE = 0.003
# 抽帧步长：每隔多少帧取一帧
FRAME_STEP = 10

# 简单距离裁剪（根据你录制的距离调整）
USE_DISTANCE_CROP = True
MIN_DIST = 0.2   # 0.2m 以内去掉（相机附近噪声）
MAX_DIST = 1.2   # 1.2m 以外去掉（背景）


def preprocess_one_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print(f"[WARN] Empty cloud: {path}")
        return None

    # 简单距离裁剪
    if USE_DISTANCE_CROP:
        pts = np.asarray(pcd.points)
        d = (pts ** 2).sum(axis=1) ** 0.5
        mask = (d > MIN_DIST) & (d < MAX_DIST)
        pcd = pcd.select_by_index([i for i, m in enumerate(mask) if m])

    # 体素降采样
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)

    return pcd


if __name__ == "__main__":
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "cloud_*.ply")))
    print(f"[INFO] Found {len(files)} frames in {INPUT_DIR}")

    selected_files = files[::FRAME_STEP]
    print(f"[INFO] Using every {FRAME_STEP} frames → {len(selected_files)} frames")

    for i, f in enumerate(selected_files):
        pcd = preprocess_one_cloud(f)
        if pcd is None or pcd.is_empty():
            continue
        out_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"[SAVE] {out_path}, points={len(pcd.points)}")

    print("[DONE] Preprocess finished.")
