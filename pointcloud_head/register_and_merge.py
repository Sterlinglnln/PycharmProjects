# register_and_merge.py
import open3d as o3d
import numpy as np
import glob
import os

INPUT_DIR = "pcd_down"
MERGED_OUTPUT = "merged_head.ply"

VOXEL_SIZE = 0.003  # 与预处理一致，用于法向估计 & ICP 搜索半径


def prepare_pcd_list():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "frame_*.ply")))
    if not files:
        raise RuntimeError(f"No point clouds found in {INPUT_DIR}")
    print(f"[INFO] Load {len(files)} frames from {INPUT_DIR}")
    pcds = []
    for f in files:
        pcd = o3d.io.read_point_cloud(f)
        if pcd.is_empty():
            print(f"[WARN] Empty pcd: {f}")
            continue
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=VOXEL_SIZE * 2.0, max_nn=30
            )
        )
        pcds.append(pcd)
    return pcds


def icp_pairwise(source, target):
    # 粗略 ICP 设置：距离阈值用几倍 voxel_size
    distance_threshold = VOXEL_SIZE * 5.0
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        ),
        init=np.eye(4)
    )
    return result.transformation, result.fitness, result.inlier_rmse


if __name__ == "__main__":
    pcds = prepare_pcd_list()

    # 累计变换矩阵：第 0 帧为世界坐标系
    transforms = [np.eye(4)]
    current_T = np.eye(4)

    for i in range(1, len(pcds)):
        print(f"\n[ICP] Register frame {i} to frame {i-1}")
        T_icp, fitness, rmse = icp_pairwise(pcds[i], pcds[i - 1])
        print(f"  fitness={fitness:.3f}, rmse={rmse:.5f}")
        current_T = current_T @ T_icp  # 累乘
        transforms.append(current_T)

    # 应用变换并融合
    merged = o3d.geometry.PointCloud()
    for i, (pcd, T) in enumerate(zip(pcds, transforms)):
        pcd_t = pcd.transform(T.copy())
        merged += pcd_t
        print(f"[MERGE] Frame {i} merged, total points={len(merged.points)}")

    # 再做一次降采样，避免点太密
    merged = merged.voxel_down_sample(VOXEL_SIZE * 0.8)

    o3d.io.write_point_cloud(MERGED_OUTPUT, merged)
    print(f"\n[DONE] Saved merged point cloud to {MERGED_OUTPUT}, points={len(merged.points)}")
