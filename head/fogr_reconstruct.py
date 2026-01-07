import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d


# -------------------------
# helpers (version safe)
# -------------------------
def _unwrap(ret):
    return ret[0] if isinstance(ret, tuple) else ret

def read_pcd(p: Path):
    pc = o3d.io.read_point_cloud(str(p))
    pc = _unwrap(pc.remove_non_finite_points())
    return pc

def estimate_normals(pcd, radius, max_nn=30):
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return pcd


# ============================================================
# 3.2 HOR + mutual NN + DTC
# ============================================================
def _global_boundary_prune(pts, q_low=0.01, q_high=0.99):
    lo = np.quantile(pts, q_low, axis=0)
    hi = np.quantile(pts, q_high, axis=0)
    m = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[m]

def hor_select_keypoints(pcd, mode, side_keep_ratio=0.60, front_keep_ratio=0.50):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] == 0:
        return pts
    pts = _global_boundary_prune(pts)

    x = pts[:, 0]
    if mode == "left":
        thr = np.quantile(x, 1.0 - side_keep_ratio)
        keep = x >= thr
    elif mode == "right":
        thr = np.quantile(x, side_keep_ratio)
        keep = x <= thr
    elif mode == "front":
        half = front_keep_ratio / 2.0
        lo = np.quantile(x, 0.5 - half)
        hi = np.quantile(x, 0.5 + half)
        keep = (x >= lo) & (x <= hi)
    else:
        raise ValueError("mode must be left/front/right")
    return pts[keep]

def _nn_1way(src, dst):
    """src每点找dst最近邻：优先scipy cKDTree，否则用Open3D KDTree（慢些）"""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        dist, idx = tree.query(src, k=1, workers=-1)
        return idx.astype(np.int32), dist.astype(np.float64)
    except Exception:
        pcd_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
        kdt = o3d.geometry.KDTreeFlann(pcd_dst)
        idx = np.empty((src.shape[0],), dtype=np.int32)
        dist = np.empty((src.shape[0],), dtype=np.float64)
        for i, p in enumerate(src):
            _, ind, d2 = kdt.search_knn_vector_3d(p, 1)
            idx[i] = int(ind[0])
            dist[i] = float(np.sqrt(d2[0]))
        return idx, dist

def mutual_nn_intersection(P, Q):
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return []
    p_to_q, _ = _nn_1way(P, Q)
    q_to_p, _ = _nn_1way(Q, P)
    pairs = []
    for i, j in enumerate(p_to_q):
        if q_to_p[j] == i:
            pairs.append((i, int(j)))
    return pairs

def dtc_filter(P, Q, pairs, sigma_k=2.0, eps_norm=1e-9):
    if len(pairs) == 0:
        return []

    pi = np.array([a for a, _ in pairs], dtype=np.int32)
    qj = np.array([b for _, b in pairs], dtype=np.int32)
    A = P[pi]
    B = Q[qj]

    d = np.linalg.norm(A - B, axis=1)
    mean_d = float(d.mean())
    std_d = float(d.std() + 1e-12)
    m_dist = (d > mean_d - sigma_k * std_d) & (d < mean_d + sigma_k * std_d)

    A2 = A[m_dist]
    B2 = B[m_dist]
    pi2 = pi[m_dist]
    qj2 = qj[m_dist]
    if A2.shape[0] == 0:
        return []

    dot = np.sum(A2 * B2, axis=1)
    na = np.linalg.norm(A2, axis=1)
    nb = np.linalg.norm(B2, axis=1)
    cosv = dot / np.maximum(na * nb, eps_norm)

    mean_c = float(cosv.mean())
    std_c = float(cosv.std() + 1e-12)
    m_cos = (cosv > mean_c - sigma_k * std_c) & (cosv < mean_c + sigma_k * std_c)

    pi3 = pi2[m_cos]
    qj3 = qj2[m_cos]
    return list(zip(pi3.tolist(), qj3.tolist()))

def rigid_fit_svd(A, B):
    """A->B 的刚体变换"""
    if A.shape[0] < 3:
        return np.eye(4, dtype=np.float64)
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def dtc_svd_init(src_pcd, tgt_pcd, min_pairs=60, side_keep_ratio=0.60, front_keep_ratio=0.50, sigma_k=2.0):
    """能成功则返回(initT, info_dict)，否则返回(None, info_dict)"""
    best = None
    for mode in ("left", "front", "right"):
        Pk = hor_select_keypoints(src_pcd, mode, side_keep_ratio=side_keep_ratio, front_keep_ratio=front_keep_ratio)
        Qk = hor_select_keypoints(tgt_pcd, mode, side_keep_ratio=side_keep_ratio, front_keep_ratio=front_keep_ratio)
        pairs0 = mutual_nn_intersection(Pk, Qk)
        pairsF = dtc_filter(Pk, Qk, pairs0, sigma_k=sigma_k)

        if len(pairsF) < min_pairs:
            cand = {"mode": mode, "pairs": len(pairsF), "status": "TOO_FEW"}
        else:
            A = Pk[[i for i, _ in pairsF]]
            B = Qk[[j for _, j in pairsF]]
            T0 = rigid_fit_svd(A, B)
            # 评分：点对多、平均距离小
            md = float(np.mean(np.linalg.norm(A - B, axis=1)))
            score = len(pairsF) - 50.0 * md
            cand = {"mode": mode, "pairs": len(pairsF), "T0": T0, "md": md, "score": score, "status": "OK"}

        if best is None or cand.get("score", -1e18) > best.get("score", -1e18):
            best = cand

    if best.get("status") != "OK":
        return None, best
    return best["T0"], best


# ============================================================
# FPFH + RANSAC fallback (for back-of-head / low overlap cases)
# ============================================================
def preprocess_down_fpfh(pcd, voxel):
    p = pcd.voxel_down_sample(float(voxel))
    p = _unwrap(p.remove_non_finite_points())
    estimate_normals(p, radius=voxel * 4.0, max_nn=30)
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 8.0, max_nn=100)
    )
    return p, f

def ransac_fpfh_init(src, tgt, voxel):
    src_d, src_f = preprocess_down_fpfh(src, voxel)
    tgt_d, tgt_f = preprocess_down_fpfh(tgt, voxel)

    dist = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, mutual_filter=True,
        max_correspondence_distance=float(dist),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(80000, 0.999)
    )
    return result.transformation, {"fitness": float(result.fitness), "rmse": float(result.inlier_rmse)}


# ============================================================
# ICP refine (multi-scale)
# ============================================================
def icp_multiscale(src, tgt, init, base_voxel=0.004, point_to_plane=True):
    scales = [
        (base_voxel * 3.0, 0.060, 40),
        (base_voxel * 2.0, 0.040, 40),
        (base_voxel * 1.0, 0.020, 60),
    ]
    T = init.copy()
    for vx, max_corr, iters in scales:
        src_d = src.voxel_down_sample(vx)
        tgt_d = tgt.voxel_down_sample(vx)
        src_d = _unwrap(src_d.remove_non_finite_points())
        tgt_d = _unwrap(tgt_d.remove_non_finite_points())

        if point_to_plane:
            estimate_normals(src_d, radius=max(0.01, vx * 6), max_nn=30)
            estimate_normals(tgt_d, radius=max(0.01, vx * 6), max_nn=30)
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d,
            max_correspondence_distance=float(max_corr),
            init=T,
            estimation_method=est,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)),
        )
        T = reg.transformation
    return T


# ============================================================
# PoseGraph optional optimization
# ============================================================
def info_matrix(src, tgt, max_corr, T):
    try:
        return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src, tgt, max_correspondence_distance=float(max_corr), transformation=T
        )
    except Exception:
        return np.eye(6, dtype=np.float64)

def optimize_pose_graph(pose_graph, max_corr=0.02):
    opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=float(max_corr),
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt,
    )
    return pose_graph


# ============================================================
# Main: pairwise chain registration -> (optional) posegraph -> fuse
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--processed", default="processed_frames")
    ap.add_argument("--out", default="recon_out")

    ap.add_argument("--base_voxel", type=float, default=0.004)
    ap.add_argument("--model_voxel", type=float, default=0.004)
    ap.add_argument("--min_pairs", type=int, default=60)

    ap.add_argument("--side_keep_ratio", type=float, default=0.60)
    ap.add_argument("--front_keep_ratio", type=float, default=0.50)
    ap.add_argument("--sigma_k", type=float, default=2.0)

    ap.add_argument("--icp_point_to_plane", type=int, default=1)

    ap.add_argument("--use_posegraph", type=int, default=1)
    ap.add_argument("--use_loop_closure", type=int, default=1)

    args = ap.parse_args()

    dataset = Path(args.dataset)
    proc_dir = dataset / args.processed
    out_dir = dataset / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(proc_dir.glob("*_processed.ply"))
    if not files:
        raise FileNotFoundError(f"No *_processed.ply in {proc_dir}")

    pcds = [read_pcd(f) for f in files]
    N = len(pcds)

    # --- chain transforms: W0=I, Wi = W_{i-1} @ T(i->i-1)
    W = [np.eye(4, dtype=np.float64)]
    pair_logs = []

    print(f"[INFO] frames={N}")

    for i in range(1, N):
        src = pcds[i]
        tgt = pcds[i - 1]

        t0 = time.time()

        # 1) try 3.2 DTC+SVD init
        init, info = dtc_svd_init(
            src, tgt,
            min_pairs=args.min_pairs,
            side_keep_ratio=args.side_keep_ratio,
            front_keep_ratio=args.front_keep_ratio,
            sigma_k=args.sigma_k,
        )

        init_name = "DTC+SVD"
        if init is None:
            # 2) fallback: FPFH+RANSAC init (very important for back-of-head)
            init_name = "FPFH+RANSAC"
            init, rinfo = ransac_fpfh_init(src, tgt, voxel=max(args.base_voxel, 0.004))
            info = {**info, **{"ransac_fitness": rinfo["fitness"], "ransac_rmse": rinfo["rmse"]}}

        # ICP refine
        T_i_to_prev = icp_multiscale(
            src, tgt, init,
            base_voxel=args.base_voxel,
            point_to_plane=bool(args.icp_point_to_plane),
        )

        # evaluate
        reg = o3d.pipelines.registration.evaluate_registration(
            src, tgt, max_correspondence_distance=0.02, transformation=T_i_to_prev
        )

        Wi = W[-1] @ T_i_to_prev  # chain
        W.append(Wi)

        sec = time.time() - t0
        pair_logs.append({
            "pair": [i, i - 1],
            "init": init_name,
            "mode": info.get("mode", None),
            "pairs": int(info.get("pairs", 0)),
            "fitness": float(reg.fitness),
            "rmse": float(reg.inlier_rmse),
            "sec": float(sec),
        })

        print(f"[{i:02d}/{N-1:02d}] {init_name:10s} "
              f"pairs={info.get('pairs',0):4d} "
              f"fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f} sec={sec:.2f}")

    # --- PoseGraph optimization (optional but recommended)
    node_poses = W
    if args.use_posegraph:
        pg = o3d.pipelines.registration.PoseGraph()
        for i in range(N):
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(node_poses[i]))

        # consecutive edges
        for i in range(1, N):
            src = pcds[i].voxel_down_sample(args.base_voxel)
            tgt = pcds[i - 1].voxel_down_sample(args.base_voxel)
            Tij = np.linalg.inv(node_poses[i - 1]) @ node_poses[i]  # i->i-1 的相对（在优化里用）
            inf = info_matrix(src, tgt, max_corr=0.03, T=Tij)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i - 1, Tij, inf, uncertain=False))

        # loop closure: last -> first
        if args.use_loop_closure and N >= 3:
            src = pcds[-1]
            tgt = pcds[0]
            init, info = dtc_svd_init(src, tgt, min_pairs=args.min_pairs,
                                      side_keep_ratio=args.side_keep_ratio,
                                      front_keep_ratio=args.front_keep_ratio,
                                      sigma_k=args.sigma_k)
            init_name = "DTC+SVD"
            if init is None:
                init_name = "FPFH+RANSAC"
                init, _ = ransac_fpfh_init(src, tgt, voxel=max(args.base_voxel, 0.004))
            T_last_to_0 = icp_multiscale(src, tgt, init, base_voxel=args.base_voxel, point_to_plane=bool(args.icp_point_to_plane))

            src_d = src.voxel_down_sample(args.base_voxel)
            tgt_d = tgt.voxel_down_sample(args.base_voxel)
            inf = info_matrix(src_d, tgt_d, max_corr=0.03, T=T_last_to_0)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(N - 1, 0, T_last_to_0, inf, uncertain=True))
            print(f"[LOOP] {init_name} added edge ({N-1} -> 0)")

        pg = optimize_pose_graph(pg, max_corr=0.03)
        node_poses = [n.pose for n in pg.nodes]

    # --- fuse all frames using optimized poses
    model = o3d.geometry.PointCloud()
    for i in range(N):
        p = o3d.geometry.PointCloud(pcds[i])
        p.transform(node_poses[i])
        model += p

    # 融合后别用“太狠的离群点删除”，否则后脑勺新区域容易被当成离群点删掉
    model = model.voxel_down_sample(args.model_voxel)
    model = _unwrap(model.remove_non_finite_points())

    model_path = out_dir / "model_fogr.ply"
    tf_path = out_dir / "transforms.json"
    log_path = out_dir / "log.json"

    o3d.io.write_point_cloud(str(model_path), model)

    tf_list = [np.asarray(T).tolist() for T in node_poses]
    tf_path.write_text(json.dumps(tf_list, indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(pair_logs, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:", model_path)
    print("Saved:", tf_path)
    print("Saved:", log_path)


if __name__ == "__main__":
    main()
