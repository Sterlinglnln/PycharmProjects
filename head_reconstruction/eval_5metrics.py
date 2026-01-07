import argparse, json, math
from pathlib import Path
import numpy as np
import open3d as o3d


def _unwrap(ret):
    return ret[0] if isinstance(ret, tuple) else ret


def load_pcd(p: Path):
    pc = o3d.io.read_point_cloud(str(p))
    pc = _unwrap(pc.remove_non_finite_points())
    if pc.is_empty():
        raise RuntimeError(f"Empty point cloud: {p}")
    return pc


def load_Ts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("transforms.json 应该是 list[4x4]（你 fogr_reconstruct.py 输出的格式）")
    Ts = [np.array(x, dtype=np.float64).reshape(4, 4) for x in data]
    return Ts


def axis_angle_to_R(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)
    return R


def rotvec_from_R(R):
    # log map -> rotation vector
    t = (np.trace(R) - 1.0) / 2.0
    t = float(np.clip(t, -1.0, 1.0))
    angle = math.acos(t)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=np.float64) / (2.0 * math.sin(angle))
    return w * angle


def euler_xyz_from_R(R):
    # intrinsic XYZ (常见做法)，输出弧度
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) < 1e-8:
        rx = math.atan2(-R[0, 1], R[1, 1])
        rz = 0.0
    else:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    return rx, ry, rz


def rmse_mae_from_dists(d):
    d = np.asarray(d, dtype=np.float64)
    rmse = math.sqrt(float(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    return rmse, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd_dir", required=True, help="processed_frames，里面是 *_processed.ply")
    ap.add_argument("--transforms", required=True, help="recon_out/transforms.json（每帧 frame->world）")
    ap.add_argument("--out", default="metrics_5.json")

    ap.add_argument("--gt_step_deg", type=float, default=18.95, help="理论相邻帧旋转角（论文约18.95°）")
    ap.add_argument("--gt_axis", default="auto", help="真值旋转轴：auto 或 x/y/z")
    ap.add_argument("--max_corr_mm", type=float, default=20.0, help="只统计距离<=该阈值的点（mm），避免背景影响")
    ap.add_argument("--use_inliers_only", type=int, default=1, help="1=只用inliers计算RMSE/MAE；0=全点")

    args = ap.parse_args()

    pcd_dir = Path(args.pcd_dir)
    pcd_paths = sorted(pcd_dir.glob("*_processed.ply"))
    if len(pcd_paths) < 2:
        raise RuntimeError("processed_frames 里至少要有2帧 *_processed.ply")

    Ts = load_Ts(Path(args.transforms))
    if len(Ts) != len(pcd_paths):
        raise RuntimeError(f"transforms({len(Ts)}) != clouds({len(pcd_paths)})，请检查排序/数量对应")

    # --- 先算每对相邻帧的估计相对旋转，用来自动估计转轴（可选）
    rel_Rs = []
    for i in range(1, len(Ts)):
        T_rel = np.linalg.inv(Ts[i - 1]) @ Ts[i]   # frame i -> frame i-1
        rel_Rs.append(T_rel[:3, :3])

    if args.gt_axis.lower() == "auto":
        rvecs = np.array([rotvec_from_R(R) for R in rel_Rs], dtype=np.float64)
        v = np.mean(rvecs, axis=0)
        if np.linalg.norm(v) < 1e-9:
            axis_gt = np.array([0, 1, 0], dtype=np.float64)
        else:
            axis_gt = v / np.linalg.norm(v)
    else:
        ax = args.gt_axis.lower()
        axis_gt = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}[ax].astype(np.float64)

    gt_step_rad = math.radians(args.gt_step_deg)
    R_gt = axis_angle_to_R(axis_gt, gt_step_rad)

    max_corr = args.max_corr_mm / 1000.0

    per_pair = []
    rmse_list = []
    mae_list = []
    rx_list = []
    ry_list = []
    rz_list = []

    for i in range(1, len(pcd_paths)):
        src = load_pcd(pcd_paths[i])
        tgt = load_pcd(pcd_paths[i - 1])

        T_rel = np.linalg.inv(Ts[i - 1]) @ Ts[i]  # i -> i-1
        src2 = o3d.geometry.PointCloud(src)
        src2.transform(T_rel)

        # 距离（最近邻）
        d = np.asarray(src2.compute_point_cloud_distance(tgt), dtype=np.float64)
        if args.use_inliers_only:
            d_used = d[d <= max_corr]
            if d_used.size < 100:
                d_used = d  # 防止过度过滤导致没点
        else:
            d_used = d

        rmse, mae = rmse_mae_from_dists(d_used)

        # 旋转误差：R_err = R_gt^{-1} * R_est
        R_est = T_rel[:3, :3]
        R_err = R_gt.T @ R_est
        rx, ry, rz = euler_xyz_from_R(R_err)  # rad

        # 取绝对误差（和表格口径一致：Rx_error/Ry_error/Rz_error）
        rx_e = abs(rx)
        ry_e = abs(ry)
        rz_e = abs(rz)

        per_pair.append({
            "pair": [i, i - 1],
            "rmse_mm": rmse * 1000.0,
            "mae_mm": mae * 1000.0,
            "Rx_error_rad": rx_e,
            "Ry_error_rad": ry_e,
            "Rz_error_rad": rz_e,
            "Rx_error_deg": math.degrees(rx_e),
            "Ry_error_deg": math.degrees(ry_e),
            "Rz_error_deg": math.degrees(rz_e),
            "used_points": int(d_used.size),
            "all_points": int(d.size),
        })

        rmse_list.append(rmse)
        mae_list.append(mae)
        rx_list.append(rx_e)
        ry_list.append(ry_e)
        rz_list.append(rz_e)

    summary = {
        # 论文表格里 RMSE/10^-3、MAE/10^-3 基本就是 mm 尺度 :contentReference[oaicite:2]{index=2}
        "RMSE_mm_mean": float(np.mean(rmse_list) * 1000.0),
        "MAE_mm_mean": float(np.mean(mae_list) * 1000.0),
        "Rx_error_mean_rad": float(np.mean(rx_list)),
        "Ry_error_mean_rad": float(np.mean(ry_list)),
        "Rz_error_mean_rad": float(np.mean(rz_list)),
        "Rx_error_mean_deg": float(np.mean([math.degrees(x) for x in rx_list])),
        "Ry_error_mean_deg": float(np.mean([math.degrees(x) for x in ry_list])),
        "Rz_error_mean_deg": float(np.mean([math.degrees(x) for x in rz_list])),
        "gt_step_deg": args.gt_step_deg,
        "gt_axis_used": axis_gt.tolist(),
        "max_corr_mm": args.max_corr_mm,
        "use_inliers_only": int(args.use_inliers_only),
        "pairs": len(per_pair),
    }

    out = {"summary": summary, "per_pair": per_pair}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
