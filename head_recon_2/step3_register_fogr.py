import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d


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


# -------------------------
# FPFH + RANSAC init
# -------------------------
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


# -------------------------
# ICP refine (multiscale)
# -------------------------
def icp_multiscale(src, tgt, init, base_voxel=0.003, point_to_plane=True):
    # 更严格的最后一层（对齐头部需要更紧）
    scales = [
        (base_voxel * 3.0, max(0.04, base_voxel * 14.0), 50),
        (base_voxel * 2.0, max(0.025, base_voxel * 10.0), 80),
        (base_voxel * 1.0, max(0.010, base_voxel * 6.0), 120),
    ]
    T = init.copy()
    for vx, max_corr, iters in scales:
        src_d = _unwrap(src.voxel_down_sample(vx).remove_non_finite_points())
        tgt_d = _unwrap(tgt.voxel_down_sample(vx).remove_non_finite_points())

        if point_to_plane:
            estimate_normals(src_d, radius=max(0.010, vx * 8.0), max_nn=30)
            estimate_normals(tgt_d, radius=max(0.010, vx * 8.0), max_nn=30)
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


def evaluate(src, tgt, T, max_corr=0.015):
    reg = o3d.pipelines.registration.evaluate_registration(
        src, tgt, max_correspondence_distance=float(max_corr), transformation=T
    )
    return float(reg.fitness), float(reg.inlier_rmse)


def transform_ok(T, max_trans=0.12, max_rot_deg=45.0):
    """
    人头相邻帧应该是小运动：平移别太大，旋转别离谱。
    """
    t = np.linalg.norm(T[:3, 3])
    R = T[:3, :3]
    # angle from trace
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(tr)))
    return (t <= float(max_trans)) and (ang <= float(max_rot_deg))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--head_dir", default="processed_frames")          # contains *_head.ply
    ap.add_argument("--rigid_dir", default="processed_frames_rigid")   # contains *_rigid.ply
    ap.add_argument("--out", default="recon_out")

    ap.add_argument("--base_voxel", type=float, default=0.003)
    ap.add_argument("--model_voxel", type=float, default=0.0025)

    ap.add_argument("--icp_point_to_plane", type=int, default=1)

    # retry window: try match to i-1 / i-2 / i-3
    ap.add_argument("--retry_k", type=int, default=3)

    # accept thresholds
    ap.add_argument("--min_fitness", type=float, default=0.25)
    ap.add_argument("--max_rmse", type=float, default=0.012)

    # posegraph
    ap.add_argument("--use_posegraph", type=int, default=1)
    ap.add_argument("--use_loop_closure", type=int, default=1)
    ap.add_argument("--local_k", type=int, default=3)

    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_dir = dataset / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    head_files = sorted((dataset / args.head_dir).glob("*_head.ply"))
    rigid_files = sorted((dataset / args.rigid_dir).glob("*_rigid.ply"))
    if not head_files:
        raise FileNotFoundError("No *_head.ply found.")
    if not rigid_files or len(rigid_files) != len(head_files):
        print("[WARN] rigid pcd missing or count mismatch, fallback: use head for registration too.")
        rigid_files = head_files

    head_pcds = [read_pcd(f) for f in head_files]    # for final fusion
    reg_pcds = [read_pcd(f) for f in rigid_files]    # for registration

    N = len(head_pcds)
    print(f"[INFO] frames={N}")

    # world poses: frame i -> frame0
    W = [np.eye(4, dtype=np.float64) for _ in range(N)]
    W[0] = np.eye(4, dtype=np.float64)

    # store best edge (i -> j)
    edges = []

    for i in range(1, N):
        t0 = time.time()

        best = None
        # try targets: i-1, i-2, ... i-retry_k
        for dj in range(1, min(int(args.retry_k), i) + 1):
            j = i - dj
            src = reg_pcds[i]
            tgt = reg_pcds[j]

            # init: FPFH+RANSAC on rigid region is usually robust
            init, rinfo = ransac_fpfh_init(src, tgt, voxel=max(args.base_voxel, 0.003))
            T = icp_multiscale(
                src, tgt, init,
                base_voxel=args.base_voxel,
                point_to_plane=bool(args.icp_point_to_plane),
            )
            fit, rmse = evaluate(src, tgt, T, max_corr=0.015)

            # plausibility check to avoid wild jumps
            ok = transform_ok(T, max_trans=0.12, max_rot_deg=45.0)

            score = fit - 8.0 * rmse  # simple score
            cand = {
                "i": i, "j": j,
                "T": T,
                "fitness": fit,
                "rmse": rmse,
                "ok": ok,
                "score": score,
                "ransac_fitness": rinfo["fitness"],
                "ransac_rmse": rinfo["rmse"],
            }

            if (best is None) or (cand["score"] > best["score"]):
                best = cand

        sec = time.time() - t0

        # accept / fallback
        if (best["fitness"] < args.min_fitness) or (best["rmse"] > args.max_rmse) or (not best["ok"]):
            print(f"[{i:02d}] FAIL-ish: best fit={best['fitness']:.3f} rmse={best['rmse']:.4f} "
                  f"(i->{best['j']})  -> still use but expect drift")
        else:
            print(f"[{i:02d}] OK     : fit={best['fitness']:.3f} rmse={best['rmse']:.4f} "
                  f"choose (i->{best['j']}) in {sec:.2f}s")

        # chain: W[i] = W[j] @ T(i->j)
        j = best["j"]
        W[i] = W[j] @ best["T"]
        edges.append(best)

    # PoseGraph
    node_poses = W
    if args.use_posegraph:
        pg = o3d.pipelines.registration.PoseGraph()
        for i in range(N):
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(node_poses[i]))

        # add chosen edges
        for e in edges:
            i, j, Tij = e["i"], e["j"], e["T"]
            src_d = reg_pcds[i].voxel_down_sample(args.base_voxel)
            tgt_d = reg_pcds[j].voxel_down_sample(args.base_voxel)
            inf = info_matrix(src_d, tgt_d, max_corr=0.03, T=Tij)
            uncertain = (abs(i - j) != 1)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, Tij, inf, uncertain=uncertain))

        # optional: add extra local loop edges (i -> i-2/i-3...)
        K = max(0, int(args.local_k))
        if K >= 2:
            for i in range(2, N):
                for j in range(max(0, i - K), i - 1):
                    src = reg_pcds[i]
                    tgt = reg_pcds[j]
                    init, _ = ransac_fpfh_init(src, tgt, voxel=max(args.base_voxel, 0.003))
                    Tij = icp_multiscale(src, tgt, init, base_voxel=args.base_voxel, point_to_plane=bool(args.icp_point_to_plane))
                    src_d = src.voxel_down_sample(args.base_voxel)
                    tgt_d = tgt.voxel_down_sample(args.base_voxel)
                    inf = info_matrix(src_d, tgt_d, max_corr=0.03, T=Tij)
                    pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j, Tij, inf, uncertain=True))
            print(f"[INFO] added local loop edges K={K}")

        # loop closure last->0 (try)
        if args.use_loop_closure and N >= 3:
            src = reg_pcds[-1]
            tgt = reg_pcds[0]
            init, _ = ransac_fpfh_init(src, tgt, voxel=max(args.base_voxel, 0.003))
            T = icp_multiscale(src, tgt, init, base_voxel=args.base_voxel, point_to_plane=bool(args.icp_point_to_plane))
            fit, rmse = evaluate(src, tgt, T, max_corr=0.02)
            if fit > 0.15:  # allow weaker loop
                src_d = src.voxel_down_sample(args.base_voxel)
                tgt_d = tgt.voxel_down_sample(args.base_voxel)
                inf = info_matrix(src_d, tgt_d, max_corr=0.03, T=T)
                pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(N - 1, 0, T, inf, uncertain=True))
                print(f"[LOOP] added edge ({N-1} -> 0) fit={fit:.3f} rmse={rmse:.4f}")
            else:
                print(f"[LOOP] skip weak closure fit={fit:.3f} rmse={rmse:.4f}")

        pg = optimize_pose_graph(pg, max_corr=0.03)
        node_poses = [n.pose for n in pg.nodes]

    # Fuse FULL head clouds using optimized poses (not rigid clouds)
    model = o3d.geometry.PointCloud()
    for i in range(N):
        p = o3d.geometry.PointCloud(head_pcds[i])
        p.transform(node_poses[i])
        model += p

    model = model.voxel_down_sample(args.model_voxel)
    model = _unwrap(model.remove_non_finite_points())

    model_path = out_dir / "model_fogr_v2.ply"
    tf_path = out_dir / "transforms_v2.json"
    log_path = out_dir / "log_v2.json"

    o3d.io.write_point_cloud(str(model_path), model)
    tf_path.write_text(json.dumps([np.asarray(T).tolist() for T in node_poses], indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(edges, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:", model_path)
    print("Saved:", tf_path)
    print("Saved:", log_path)


if __name__ == "__main__":
    main()
