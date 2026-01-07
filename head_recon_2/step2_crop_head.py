import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def median_depth_in_patch(depth_u16: np.ndarray, u: int, v: int, radii=(2, 4, 6, 8, 10, 14)):
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
    x = (u - intr["cx"]) * z_m / intr["fx"]
    y = (v - intr["cy"]) * z_m / intr["fy"]
    return np.array([x, y, z_m], dtype=np.float64)


def make_pcd_open3d(color_bgr: np.ndarray, depth_u16: np.ndarray, intr, depth_trunc=1.6):
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(depth_u16)

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


def voxel_down(pcd: o3d.geometry.PointCloud, voxel):
    if len(pcd.points) == 0:
        return pcd
    return pcd.voxel_down_sample(float(voxel))


def gentle_denoise_hair_friendly(pcd: o3d.geometry.PointCloud, nb=8, radius=0.012):
    if len(pcd.points) < 2000:
        return pcd
    pcd2, _ = pcd.remove_radius_outlier(nb_points=int(nb), radius=float(radius))
    return pcd2


# -----------------------------
# MediaPipe helpers
# -----------------------------
def get_face_bbox_fd(fd, color_bgr):
    h, w = color_bgr.shape[:2]
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


def _conn_to_indices(connections):
    s = set()
    for a, b in connections:
        s.add(int(a))
        s.add(int(b))
    return sorted(s)


_NOSE_IDXS = _conn_to_indices(mp.solutions.face_mesh.FACEMESH_NOSE)
_OVAL_IDXS = _conn_to_indices(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)


def get_nose_chin_uv_and_oval_pts(fm, color_bgr, depth_u16, intr,
                                 z_min=0.30, z_max=1.60,
                                 max_chin_delta_z=0.25):
    """
    return (nose_uv, chin_uv, oval_uv_pts, dbg)
    """
    h, w = depth_u16.shape
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None, {"src": "NO_FACE"}

    lms = res.multi_face_landmarks[0].landmark

    # oval pts (unordered), later use convex hull
    oval_uv = []
    for idx in _OVAL_IDXS:
        lm = lms[idx]
        u = clamp(int(lm.x * w), 0, w - 1)
        v = clamp(int(lm.y * h), 0, h - 1)
        oval_uv.append((u, v))

    # nose: pick nearest depth in NOSE region
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
        return None, None, oval_uv, {"src": "NOSE_FAIL"}

    nose_cands.sort(key=lambda t: t[0])
    z_n, u_n, v_n = nose_cands[0]

    # chin: from OVAL points with depth constraint
    oval_pts3 = []
    for (u, v) in oval_uv:
        d = median_depth_in_patch(depth_u16, u, v)
        if d <= 0:
            continue
        z = d * intr["depth_scale"]
        if z_min <= z <= z_max:
            oval_pts3.append((u, v, z))

    if len(oval_pts3) < 20:
        return (u_n, v_n), None, oval_uv, {"src": "CHIN_FAIL", "nose_uvz": (int(u_n), int(v_n), float(z_n))}

    oval3 = np.array(oval_pts3, dtype=np.float64)  # u,v,z
    u_center = float(np.median(oval3[:, 0]))
    face_w = float(np.max(oval3[:, 0]) - np.min(oval3[:, 0]) + 1e-6)
    v_bottom = float(np.max(oval3[:, 1]))

    bottom_band = 0.10 * h
    m_bottom = oval3[:, 1] >= (v_bottom - bottom_band)
    m_center = np.abs(oval3[:, 0] - u_center) <= (0.25 * face_w)
    m_depth = oval3[:, 2] <= (z_n + max_chin_delta_z)  # avoid neck/cloth

    mask = m_bottom & m_center & m_depth
    if int(mask.sum()) < 5:
        mask = m_bottom & m_depth
    if int(mask.sum()) < 5:
        mask = m_bottom

    cand = oval3[mask]
    i = int(np.argmax(cand[:, 1]))
    u_c, v_c, z_c = cand[i]

    if int(v_c) <= int(v_n) + 8:
        return (int(u_n), int(v_n)), None, oval_uv, {"src": "CHIN_SANITY_FAIL", "nose_uvz": (int(u_n), int(v_n), float(z_n))}

    dbg = {
        "src": "FaceMeshDepth+Region",
        "nose_uvz": (int(u_n), int(v_n), float(z_n)),
        "chin_uvz": (int(u_c), int(v_c), float(z_c)),
    }
    return (int(u_n), int(v_n)), (int(u_c), int(v_c)), oval_uv, dbg


# -----------------------------
# Masks
# -----------------------------
def oval_convex_mask(h, w, oval_uv, dilate_iter=5):
    """
    rigid mask: convex hull of face oval, then dilate a bit.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not oval_uv or len(oval_uv) < 6:
        return mask
    pts = np.array(oval_uv, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, k, iterations=int(dilate_iter))
    return mask


def head_ellipse_mask(h, w, face_bbox, extra_up=0.75, extra_down=1.10, extra_lr=0.25):
    """
    head mask: from face bbox ellipse expanded upwards for hair, downwards slightly.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is None:
        return mask
    x, y, bw, bh = face_bbox
    x0 = clamp(int(x - extra_lr * bw), 0, w - 1)
    x1 = clamp(int(x + (1.0 + extra_lr) * bw), 0, w - 1)
    y0 = clamp(int(y - extra_up * bh), 0, h - 1)
    y1 = clamp(int(y + extra_down * bh), 0, h - 1)
    cx = int((x0 + x1) * 0.5)
    cy = int((y0 + y1) * 0.5)
    ax = max(1, int((x1 - x0) * 0.50))
    ay = max(1, int((y1 - y0) * 0.52))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def apply_mask_to_depth(depth_u16, mask_u8):
    d = depth_u16.copy()
    d[mask_u8 == 0] = 0
    return d


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", default="subset19.txt")

    ap.add_argument("--out_head", default="processed_frames")
    ap.add_argument("--out_rigid", default="processed_frames_rigid")
    ap.add_argument("--mask_out", default="head_masks")
    ap.add_argument("--dbg_out", default="step2_debug")

    ap.add_argument("--depth_trunc", type=float, default=1.6)
    ap.add_argument("--coarse_voxel", type=float, default=0.004)
    ap.add_argument("--head_voxel", type=float, default=0.0025)   # keep hair
    ap.add_argument("--rigid_voxel", type=float, default=0.0030)  # stable alignment

    ap.add_argument("--max_chin_delta_z", type=float, default=0.22)

    # denoise
    ap.add_argument("--radius_nb", type=int, default=8)
    ap.add_argument("--radius_m", type=float, default=0.012)

    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    frames_dir = dataset_dir / "frames"
    intr = json.loads((dataset_dir / "intrinsics.json").read_text(encoding="utf-8"))

    subset_path = dataset_dir / args.subset
    idx_list = [int(x.strip()) for x in subset_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    out_head = dataset_dir / args.out_head
    out_rigid = dataset_dir / args.out_rigid
    mask_dir = dataset_dir / args.mask_out
    dbg_dir = dataset_dir / args.dbg_out
    for d in (out_head, out_rigid, mask_dir, dbg_dir):
        d.mkdir(parents=True, exist_ok=True)

    mp_fd = mp.solutions.face_detection
    mp_fm = mp.solutions.face_mesh
    fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    fm = mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    try:
        for k, idx in enumerate(idx_list):
            color_path = frames_dir / f"{idx:04d}_color.png"
            depth_path = frames_dir / f"{idx:04d}_depth.png"
            color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if color_bgr is None or depth_u16 is None:
                print(f"[WARN] missing frame idx={idx}")
                continue
            h, w = depth_u16.shape

            face_bbox = get_face_bbox_fd(fd, color_bgr)

            nose_uv, chin_uv, oval_uv, dbg = get_nose_chin_uv_and_oval_pts(
                fm, color_bgr, depth_u16, intr,
                z_min=0.30, z_max=float(args.depth_trunc),
                max_chin_delta_z=args.max_chin_delta_z
            )

            # rigid mask: face oval convex hull
            rigid_mask = oval_convex_mask(h, w, oval_uv, dilate_iter=5)

            # cut off below chin a bit (avoid neck/cloth in rigid mask)
            if chin_uv is not None:
                _, v_c = chin_uv
                y_cut = clamp(int(v_c + 0.04 * h), 0, h - 1)
                rigid_mask[y_cut:, :] = 0

            # head mask: bbox ellipse (keep hair)
            head_mask = head_ellipse_mask(h, w, face_bbox)

            # If bbox missing, fallback: use rigid mask dilated as head mask
            if face_bbox is None or int(head_mask.sum()) == 0:
                head_mask = rigid_mask.copy()
                if int(head_mask.sum()) > 0:
                    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                    head_mask = cv2.dilate(head_mask, k2, iterations=2)

            # Create masked depths
            depth_rigid = apply_mask_to_depth(depth_u16, rigid_mask)
            depth_head = apply_mask_to_depth(depth_u16, head_mask)

            # Build point clouds
            pcd_rigid = make_pcd_open3d(color_bgr, depth_rigid, intr, depth_trunc=args.depth_trunc)
            pcd_head = make_pcd_open3d(color_bgr, depth_head, intr, depth_trunc=args.depth_trunc)

            # Downsample
            pcd_rigid = voxel_down(pcd_rigid, args.rigid_voxel)
            pcd_head = voxel_down(pcd_head, args.head_voxel)

            # Gentle denoise (hair-friendly) only for head cloud; rigid keep clean but not too aggressive
            pcd_head = gentle_denoise_hair_friendly(pcd_head, nb=args.radius_nb, radius=args.radius_m)

            # Save
            o3d.io.write_point_cloud(str(out_head / f"{k:04d}_head.ply"), pcd_head)
            o3d.io.write_point_cloud(str(out_rigid / f"{k:04d}_rigid.ply"), pcd_rigid)
            cv2.imwrite(str(mask_dir / f"{k:04d}_mask.png"), head_mask)

            # Debug
            vis = color_bgr.copy()
            cv2.putText(vis, f"k={k:03d} idx={idx:04d}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if face_bbox is not None:
                x, y, bw, bh = face_bbox
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
            if nose_uv is not None:
                cv2.circle(vis, nose_uv, 6, (0, 255, 0), -1)
            if chin_uv is not None:
                cv2.circle(vis, chin_uv, 6, (0, 0, 255), -1)
            # overlay masks
            overlay = vis.copy()
            overlay[head_mask > 0] = (overlay[head_mask > 0] * 0.6 + np.array([30, 200, 30]) * 0.4).astype(np.uint8)
            overlay[rigid_mask > 0] = (overlay[rigid_mask > 0] * 0.6 + np.array([200, 30, 30]) * 0.4).astype(np.uint8)
            cv2.imwrite(str(dbg_dir / f"{k:04d}_mask.png"), overlay)

            print(f"[{k+1:02d}/{len(idx_list):02d}] head_pts={len(pcd_head.points)} rigid_pts={len(pcd_rigid.points)}")

    finally:
        fd.close()
        fm.close()

    print("Done.")
    print("Head PCD :", out_head)
    print("Rigid PCD:", out_rigid)
    print("Masks    :", mask_dir)
    print("Debug    :", dbg_dir)


if __name__ == "__main__":
    main()
