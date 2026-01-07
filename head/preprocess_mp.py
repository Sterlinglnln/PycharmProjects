import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp


# =========================
# Utils
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def median_depth_in_patch(depth_u16: np.ndarray, u: int, v: int, radii=(2, 4, 6, 8, 10, 14)):
    """关键点深度取邻域median，避免0洞"""
    h, w = depth_u16.shape
    for r in radii:
        x0, x1 = clamp(u - r, 0, w - 1), clamp(u + r, 0, w - 1)
        y0, y1 = clamp(v - r, 0, h - 1), clamp(v + r, 0, h - 1)
        patch = depth_u16[y0:y1 + 1, x0:x1 + 1]
        vals = patch[patch > 0]
        if vals.size > 0:
            return float(np.median(vals))
    return 0.0


def pixel_to_xyz(u: int, v: int, z_m: float, intr):
    """相机坐标系（Open3D）：x右 y下 z前"""
    x = (u - intr["cx"]) * z_m / intr["fx"]
    y = (v - intr["cy"]) * z_m / intr["fy"]
    return np.array([x, y, z_m], dtype=np.float64)


# =========================
# Point cloud build + voxel filter
# =========================
def make_pcd_open3d(color_bgr: np.ndarray, depth_u16: np.ndarray, intr, depth_trunc=1.6):
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(depth_u16)

    # depth(m) = depth_u16 * intr["depth_scale"]  -> Open3D depth_scale = 1/depth_scale
    depth_scale_o3d = float(1.0 / intr["depth_scale"])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=depth_scale_o3d,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False
    )

    pin = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pin)
    return pcd


def voxel_filter(pcd: o3d.geometry.PointCloud, voxel=0.004):
    pcd = pcd.voxel_down_sample(float(voxel))
    if len(pcd.points) > 2000:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


# =========================
# Face bbox (for auto thresholds / fallback center)
# =========================
def get_face_bbox(color_bgr):
    mp_fd = mp.solutions.face_detection
    h, w = color_bgr.shape[:2]
    with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if not res.detections:
            return None
        det = res.detections[0]
        bb = det.location_data.relative_bounding_box
        x = clamp(int(bb.xmin * w), 0, w - 1)
        y = clamp(int(bb.ymin * h), 0, h - 1)
        bw = clamp(int(bb.width * w), 1, w - x)
        bh = clamp(int(bb.height * h), 1, h - y)
        return (x, y, bw, bh)


# =========================
# FaceMesh region-constrained keypoints (robust)
#   - nose: only FACEMESH_NOSE region, pick nearest in depth
#   - chin: only FACE_OVAL bottom band + near face center + depth constraint
# =========================
def _conn_to_indices(connections):
    s = set()
    for a, b in connections:
        s.add(int(a))
        s.add(int(b))
    return sorted(s)


_NOSE_IDXS = _conn_to_indices(mp.solutions.face_mesh.FACEMESH_NOSE)
_OVAL_IDXS = _conn_to_indices(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)


def get_nose_chin_uv(color_bgr, depth_u16, intr,
                     z_min=0.30, z_max=1.60,
                     max_chin_delta_z=0.25):
    """
    返回 (nose_uv, chin_uv, dbg)
    """
    h, w = depth_u16.shape
    mp_fm = mp.solutions.face_mesh

    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as fm:
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, None, {"src": "NO_FACE"}

        lms = res.multi_face_landmarks[0].landmark

        # ---- nose: only NOSE region, nearest in depth ----
        nose_cands = []
        for idx in _NOSE_IDXS:
            lm = lms[idx]
            u = clamp(int(lm.x * w), 0, w - 1)
            v = clamp(int(lm.y * h), 0, h - 1)
            d = median_depth_in_patch(depth_u16, u, v)
            if d <= 0:
                continue
            z = d * intr["depth_scale"]
            if z_min <= z <= z_max:
                nose_cands.append((z, u, v))

        if len(nose_cands) < 5:
            return None, None, {"src": "NOSE_FAIL"}

        nose_cands.sort(key=lambda t: t[0])
        z_n, u_n, v_n = nose_cands[0]

        # ---- chin: only FACE_OVAL region ----
        oval_pts = []
        for idx in _OVAL_IDXS:
            lm = lms[idx]
            u = clamp(int(lm.x * w), 0, w - 1)
            v = clamp(int(lm.y * h), 0, h - 1)
            d = median_depth_in_patch(depth_u16, u, v)
            if d <= 0:
                continue
            z = d * intr["depth_scale"]
            if z_min <= z <= z_max:
                oval_pts.append((u, v, z))

        if len(oval_pts) < 20:
            return (u_n, v_n), None, {"src": "CHIN_FAIL", "nose_uvz": (int(u_n), int(v_n), float(z_n))}

        oval = np.array(oval_pts, dtype=np.float64)  # u,v,z
        u_center = float(np.median(oval[:, 0]))
        face_w = float(np.max(oval[:, 0]) - np.min(oval[:, 0]) + 1e-6)
        v_bottom = float(np.max(oval[:, 1]))

        bottom_band = 0.10 * h
        m_bottom = oval[:, 1] >= (v_bottom - bottom_band)
        m_center = np.abs(oval[:, 0] - u_center) <= (0.25 * face_w)
        m_depth = oval[:, 2] <= (z_n + max_chin_delta_z)  # 防止滑到脖子/衣服

        mask = m_bottom & m_center & m_depth
        if int(mask.sum()) < 5:
            mask = m_bottom & m_depth
        if int(mask.sum()) < 5:
            mask = m_bottom

        cand = oval[mask]
        i = int(np.argmax(cand[:, 1]))  # v 最大 -> 最靠下
        u_c, v_c, z_c = cand[i]

        # 基本合理性检查：下巴应比鼻尖更“靠下”（v 更大）
        if int(v_c) <= int(v_n) + 8:
            return (int(u_n), int(v_n)), None, {"src": "CHIN_SANITY_FAIL", "nose_uvz": (int(u_n), int(v_n), float(z_n))}

        dbg = {
            "src": "FaceMeshDepth+Region",
            "nose_uvz": (int(u_n), int(v_n), float(z_n)),
            "chin_uvz": (int(u_c), int(v_c), float(z_c)),
        }
        return (int(u_n), int(v_n)), (int(u_c), int(v_c)), dbg


# =========================
# Thresholds (paper eps_z, eps_y1, eps_y2)
# =========================
def compute_thresholds_auto(intr, z_nose, face_bbox,
                            epsz_min=0.20, epsz_max=0.42,
                            ey1_min=0.18, ey1_max=0.45,
                            ey2_min=0.06, ey2_max=0.20):
    """
    用 face bbox 宽度 + z_nose 估计脸宽(m)： face_w_m ≈ bbox_w / fx * z
    再映射到阈值范围（经验映射，可稳健用在不同距离）
    """
    if face_bbox is None:
        return 0.30, 0.30, 0.12

    _, _, bw, _ = face_bbox
    face_w_m = (float(bw) * float(z_nose)) / float(intr["fx"])  # meters

    eps_z = 1.20 * face_w_m
    eps_y1 = 1.60 * face_w_m
    eps_y2 = 0.35 * face_w_m

    eps_z = float(np.clip(eps_z, epsz_min, epsz_max))
    eps_y1 = float(np.clip(eps_y1, ey1_min, ey1_max))
    eps_y2 = float(np.clip(eps_y2, ey2_min, ey2_max))
    return eps_z, eps_y1, eps_y2


# =========================
# Paper axial split (Eq.1, Eq.2)
# =========================
def axial_split(pcd: o3d.geometry.PointCloud, pnose: np.ndarray, pchin: np.ndarray,
                eps_z: float, eps_y1: float, eps_y2: float):
    if len(pcd.points) == 0:
        return pcd

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) == len(pcd.points) else None

    # Eq.(1): PC_z = { Xi in PC | z_i < Pnose^z + eps_z }
    m_z = pts[:, 2] < (pnose[2] + eps_z)
    pts_z = pts[m_z]
    cols_z = cols[m_z] if cols is not None else None
    if pts_z.shape[0] == 0:
        return pcd

    # Eq.(2): P_nc = { Xi in PC_z | Pchin^y - eps_y1 < y_i < Pchin^y + eps_y2 }
    m_y = (pts_z[:, 1] > (pchin[1] - eps_y1)) & (pts_z[:, 1] < (pchin[1] + eps_y2))
    pts_f = pts_z[m_y]
    cols_f = cols_z[m_y] if cols_z is not None else None

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_f)
    if cols_f is not None:
        out.colors = o3d.utility.Vector3dVector(cols_f)
    return out


# =========================
# Fallback for back-head: sphere crop around head center
# =========================
def sphere_crop(pcd: o3d.geometry.PointCloud, center_xyz: np.ndarray, r: float):
    if len(pcd.points) == 0:
        return pcd
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) == len(pcd.points) else None
    d = np.linalg.norm(pts - center_xyz[None, :], axis=1)
    m = d < float(r)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[m])
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols[m])
    return out


def approx_center_from_bbox(depth_u16, intr, bbox):
    """当无脸关键点但有bbox时，用bbox中心+深度估计一个3D中心（用于球裁剪兜底）"""
    if bbox is None:
        return None
    x, y, bw, bh = bbox
    u = int(x + bw * 0.5)
    v = int(y + bh * 0.55)  # 略偏下更接近头部中部
    d = median_depth_in_patch(depth_u16, u, v)
    if d <= 0:
        return None
    z = d * intr["depth_scale"]
    return pixel_to_xyz(u, v, z, intr)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\Users\77646\PycharmProjects\head_reconstruction\dataset",
                    help="dataset dir (frames/, intrinsics.json, subset19.txt)")
    ap.add_argument("--subset", default="subset19.txt", help="subset file name in dataset dir")
    ap.add_argument("--out", default="processed_frames", help="output folder name under dataset dir")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--depth_trunc", type=float, default=1.6)

    # 论文阈值：可固定，也可自动
    ap.add_argument("--fixed_eps_z", type=float, default=-1.0)
    ap.add_argument("--fixed_eps_y1", type=float, default=-1.0)
    ap.add_argument("--fixed_eps_y2", type=float, default=-1.0)

    # 下巴深度约束（越小越不容易滑到脖子）
    ap.add_argument("--max_chin_delta_z", type=float, default=0.25)

    # 背面/无脸帧：球形裁剪半径（越大越保后脑勺，但肩膀也更容易进来）
    ap.add_argument("--fallback_sphere_r", type=float, default=0.28)

    # 是否在“正面公式裁剪后”再额外做一次球形去肩膀（默认关闭，确保不误切后脑勺）
    ap.add_argument("--post_sphere_r", type=float, default=0.0)

    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    frames_dir = dataset_dir / "frames"
    intr = json.loads((dataset_dir / "intrinsics.json").read_text(encoding="utf-8"))

    subset_path = dataset_dir / args.subset
    idx_list = [int(x.strip()) for x in subset_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    out_dir = dataset_dir / args.out
    dbg_dir = dataset_dir / (args.out + "_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    # 记录上一帧头部中心（用于背面帧保后脑勺）
    head_center = None

    for k, idx in enumerate(idx_list):
        color_path = frames_dir / f"{idx:04d}_color.png"
        depth_path = frames_dir / f"{idx:04d}_depth.png"

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if color_bgr is None or depth_u16 is None:
            print(f"[WARN] missing frame idx={idx}")
            continue

        face_bbox = get_face_bbox(color_bgr)

        # 先构建点云 + 体素滤波（论文中的“体素滤波”在这里）
        pcd = make_pcd_open3d(color_bgr, depth_u16, intr, depth_trunc=args.depth_trunc)
        pcd = voxel_filter(pcd, voxel=args.voxel)

        # 尝试获取 nose/chin
        nose_uv, chin_uv, dbg = get_nose_chin_uv(
            color_bgr, depth_u16, intr,
            z_min=0.30, z_max=float(args.depth_trunc),
            max_chin_delta_z=args.max_chin_delta_z
        )

        # 是否能按论文公式裁剪
        use_paper_split = (nose_uv is not None and chin_uv is not None)

        if use_paper_split:
            # 2D -> 3D nose/chin
            u_n, v_n = nose_uv
            u_c, v_c = chin_uv

            d_n = median_depth_in_patch(depth_u16, u_n, v_n)
            d_c = median_depth_in_patch(depth_u16, u_c, v_c)

            if d_n <= 0 or d_c <= 0:
                use_paper_split = False
                dbg["src"] = dbg.get("src", "") + "+DEPTH_FAIL"

        if use_paper_split:
            z_n = d_n * intr["depth_scale"]
            z_c = d_c * intr["depth_scale"]
            pnose = pixel_to_xyz(u_n, v_n, z_n, intr)
            pchin = pixel_to_xyz(u_c, v_c, z_c, intr)

            # 阈值
            if args.fixed_eps_z > 0 and args.fixed_eps_y1 > 0 and args.fixed_eps_y2 > 0:
                eps_z, eps_y1, eps_y2 = args.fixed_eps_z, args.fixed_eps_y1, args.fixed_eps_y2
                dbg["eps_mode"] = "fixed"
            else:
                eps_z, eps_y1, eps_y2 = compute_thresholds_auto(intr, z_n, face_bbox)
                dbg["eps_mode"] = "auto"

            # 论文公式(1)(2)轴向分割
            pcd2 = axial_split(pcd, pnose, pchin, eps_z, eps_y1, eps_y2)

            # 更新头部中心（供背面帧用）
            head_center = 0.5 * (pnose + pchin)

            # 可选：正面裁剪后再做一个球裁剪去肩膀（默认关闭）
            if args.post_sphere_r and args.post_sphere_r > 0.0:
                pcd2 = sphere_crop(pcd2, head_center, r=float(args.post_sphere_r))

            # 再轻量去噪
            if len(pcd2.points) > 2000:
                pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            src_tag = "PAPER_SPLIT"

        else:
            # 背面/无脸/关键点不可信：保后脑勺策略
            # 1) 如果已有 head_center，用它球形裁剪
            # 2) 否则如果有 bbox，用 bbox 中心估一个 head_center
            # 3) 再否则不裁剪（至少不中断）
            center = head_center
            if center is None:
                center = approx_center_from_bbox(depth_u16, intr, face_bbox)
                if center is not None:
                    head_center = center  # 初始化中心，后续背面帧也能用

            if center is not None:
                pcd2 = sphere_crop(pcd, center, r=float(args.fallback_sphere_r))
                if len(pcd2.points) > 2000:
                    pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                src_tag = "SPHERE_FALLBACK"
            else:
                pcd2 = pcd
                src_tag = "NO_SPLIT_RAW"

        # 保存
        o3d.io.write_point_cloud(str(out_dir / f"{k:04d}_processed.ply"), pcd2)

        # Debug overlay
        vis = color_bgr.copy()
        cv2.putText(vis, f"mode={src_tag}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if face_bbox is not None:
            x, y, bw, bh = face_bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 255, 0), 2)

        if "nose_uvz" in dbg:
            u_n, v_n, z_n = dbg["nose_uvz"]
            cv2.circle(vis, (int(u_n), int(v_n)), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"nose({u_n},{v_n}) z={z_n:.3f}m", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if "chin_uvz" in dbg:
            u_c, v_c, z_c = dbg["chin_uvz"]
            cv2.circle(vis, (int(u_c), int(v_c)), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"chin({u_c},{v_c}) z={z_c:.3f}m", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if use_paper_split:
            cv2.putText(vis, f"eps_z={eps_z:.3f} eps_y1={eps_y1:.3f} eps_y2={eps_y2:.3f}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            if head_center is not None:
                cv2.putText(vis, f"fallback_sphere_r={args.fallback_sphere_r:.2f}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imwrite(str(dbg_dir / f"{k:04d}_kp.png"), vis)

        print(f"[{k+1:02d}/{len(idx_list):02d}] idx={idx} mode={src_tag} pts={len(pcd2.points)}")

    print("Done.")
    print("Processed:", out_dir)
    print("Debug    :", dbg_dir)


if __name__ == "__main__":
    main()
