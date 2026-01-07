import argparse
import numpy as np
import open3d as o3d


def surface_variation(pcd, radius=0.01):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    sv = np.zeros(len(pts), dtype=np.float32)

    for i, p in enumerate(pts):
        _, idx, _ = kdt.search_radius_vector_3d(p, float(radius))
        if len(idx) < 12:
            continue
        Q = pts[idx] - pts[idx].mean(axis=0)
        C = (Q.T @ Q) / max(1, Q.shape[0])
        w = np.linalg.eigvalsh(C)
        sv[i] = float(w[0] / (w.sum() + 1e-12))
    return sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", required=True, help="input point cloud ply (e.g., recon_out/tsdf_model.ply)")
    ap.add_argument("--out_ply", required=True, help="output ply with grayscale colors")
    ap.add_argument("--radius", type=float, default=0.01)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.in_ply)
    sv = surface_variation(pcd, radius=args.radius)

    if sv.size == 0:
        print("[WARN] empty point cloud")
        o3d.io.write_point_cloud(args.out_ply, pcd)
        return

    sv_n = (sv - sv.min()) / (sv.max() - sv.min() + 1e-12)
    colors = np.stack([sv_n, sv_n, sv_n], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(args.out_ply, pcd)
    print("Saved:", args.out_ply)
    print(f"SV stats: min={sv.min():.6f} max={sv.max():.6f} mean={sv.mean():.6f}")


if __name__ == "__main__":
    main()
