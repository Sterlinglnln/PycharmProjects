import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d


def load_intrinsics(intr_path: Path):
    intr = json.loads(intr_path.read_text(encoding="utf-8"))
    pin = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"],
        intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    )
    depth_scale_o3d = float(1.0 / intr["depth_scale"])  # e.g. 1/0.001 = 1000
    return intr, pin, depth_scale_o3d


def prep_pcd_for_mesh(pcd: o3d.geometry.PointCloud, voxel=0.003):
    # 轻度降采样 + 去噪 + 法向
    pcd = pcd.voxel_down_sample(float(voxel))
    if len(pcd.points) > 3000:
        ret = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        pcd = ret[0] if isinstance(ret, tuple) else ret

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.01, voxel * 6), max_nn=30)
    )
    # 让法向方向更一致（Poisson/BPA更稳）
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass
    return pcd


def mesh_poisson(pcd, depth=10, density_q=0.02):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    densities = np.asarray(densities)
    thr = np.quantile(densities, float(density_q))
    mesh = mesh.remove_vertices_by_mask(densities < thr)
    mesh.compute_vertex_normals()
    return mesh


def mesh_bpa(pcd, radii=None):
    # radii 自动估计（基于平均最近邻距离）
    if radii is None:
        d = pcd.compute_nearest_neighbor_distance()
        d = float(np.mean(d)) if len(d) else 0.005
        radii = [d * 1.5, d * 2.5, d * 3.5]
    radii = o3d.utility.DoubleVector([float(r) for r in radii])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.compute_vertex_normals()
    return mesh


def mesh_tsdf(dataset_dir: Path, out_mesh: Path,
              frames_dir_name="frames",
              subset_name="subset19.txt",
              transforms_name="recon_out/transforms.json",
              intr_name="intrinsics.json",
              voxel=0.004, trunc_mul=5.0,
              depth_trunc=1.6):
    """
    用 subset19.txt 对应的原始 RGB/Depth 图 + transforms.json 做 TSDF 融合。
    关键点：Open3D TSDF integrate 的 extrinsic 是 world->camera，所以要用 inv(T)
    这里 T 是你配准得到的 camera->world（把该帧点云变到模型坐标系）
    """
    intr, pin, depth_scale_o3d = load_intrinsics(dataset_dir / intr_name)

    subset = [int(x.strip()) for x in (dataset_dir / subset_name).read_text(encoding="utf-8").splitlines() if x.strip()]
    Ts = json.loads((dataset_dir / transforms_name).read_text(encoding="utf-8"))
    Ts = [np.array(t, dtype=np.float64) for t in Ts]

    if len(Ts) != len(subset):
        raise RuntimeError(f"transforms({len(Ts)}) != subset({len(subset)}). 请确认 transforms.json 对应的是 subset19 的帧数。")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel),
        sdf_trunc=float(trunc_mul * voxel),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    frames_dir = dataset_dir / frames_dir_name

    for k, idx in enumerate(subset):
        color_path = frames_dir / f"{idx:04d}_color.png"
        depth_path = frames_dir / f"{idx:04d}_depth.png"
        if not color_path.exists() or not depth_path.exists():
            print(f"[WARN] missing {idx:04d} color/depth, skip")
            continue

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_u16 is None:
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_u16)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=float(depth_scale_o3d),
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False
        )

        T_c2w = Ts[k]
        extrinsic_w2c = np.linalg.inv(T_c2w)   # 关键：TSDF integrate需要 world->camera
        volume.integrate(rgbd, pin, extrinsic_w2c)

        if (k + 1) % 5 == 0:
            print(f"TSDF integrated {k+1}/{len(subset)}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print("Saved:", out_mesh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\Users\77646\PycharmProjects\head_reconstruction\dataset")

    ap.add_argument("--method", choices=["poisson", "bpa", "tsdf"], default="poisson")
    ap.add_argument("--in_ply", default="recon_out/model_fogr.ply")
    ap.add_argument("--out_mesh", default="recon_out/model_mesh.ply")

    # common
    ap.add_argument("--voxel", type=float, default=0.003)

    # poisson
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--density_q", type=float, default=0.02)

    # bpa
    ap.add_argument("--bpa_radii", type=str, default="")  # e.g. "0.006,0.010,0.014"

    # tsdf
    ap.add_argument("--subset", default="subset19.txt")
    ap.add_argument("--frames_dir", default="frames")
    ap.add_argument("--transforms", default="recon_out/transforms.json")
    ap.add_argument("--tsdf_voxel", type=float, default=0.004)
    ap.add_argument("--trunc_mul", type=float, default=5.0)
    ap.add_argument("--depth_trunc", type=float, default=1.6)

    args = ap.parse_args()
    dataset_dir = Path(args.dataset)
    out_mesh = dataset_dir / args.out_mesh

    if args.method == "tsdf":
        mesh_tsdf(
            dataset_dir=dataset_dir,
            out_mesh=out_mesh,
            frames_dir_name=args.frames_dir,
            subset_name=args.subset,
            transforms_name=args.transforms,
            voxel=args.tsdf_voxel,
            trunc_mul=args.trunc_mul,
            depth_trunc=args.depth_trunc
        )
        return

    # poisson / bpa from fused point cloud
    in_ply = dataset_dir / args.in_ply
    pcd = o3d.io.read_point_cloud(str(in_ply))
    if len(pcd.points) == 0:
        raise RuntimeError("Input point cloud empty: " + str(in_ply))

    pcd = prep_pcd_for_mesh(pcd, voxel=args.voxel)

    if args.method == "poisson":
        mesh = mesh_poisson(pcd, depth=args.poisson_depth, density_q=args.density_q)
    else:
        radii = None
        if args.bpa_radii.strip():
            radii = [float(x) for x in args.bpa_radii.split(",")]
        mesh = mesh_bpa(pcd, radii=radii)

    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print("Saved:", out_mesh)


if __name__ == "__main__":
    main()
