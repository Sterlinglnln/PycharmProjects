# TSDF 融合生成高质量 mesh
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_intr(dataset_dir: Path):
    intr = json.loads((dataset_dir / "intrinsics.json").read_text(encoding="utf-8"))
    pin = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    )
    depth_scale_o3d = float(1.0 / intr["depth_scale"])
    return intr, pin, depth_scale_o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--masks", default="head_masks")
    ap.add_argument("--recon", default="recon_out")

    ap.add_argument("--transforms", default="transforms.json")
    ap.add_argument("--voxel", type=float, default=0.0025)  # 2.5mm
    ap.add_argument("--trunc", type=float, default=0.01)     # ~4x voxel
    ap.add_argument("--depth_trunc", type=float, default=1.6)
    ap.add_argument("--use_mask", type=int, default=1)

    args = ap.parse_args()

    dataset = Path(args.dataset)
    frames_dir = dataset / args.frames
    masks_dir = dataset / args.masks
    out_dir = dataset / args.recon
    out_dir.mkdir(parents=True, exist_ok=True)

    intr, pin, depth_scale_o3d = load_intr(dataset)

    poses = json.loads((out_dir / args.transforms).read_text(encoding="utf-8"))
    poses = [np.array(T, dtype=np.float64) for T in poses]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel),
        sdf_trunc=float(args.trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, T_cw in enumerate(poses):
        color_path = frames_dir / f"{i:04d}_color.png"
        depth_path = frames_dir / f"{i:04d}_depth.png"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            continue

        if args.use_mask:
            mask_path = masks_dir / f"{i:04d}_mask.png"
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                depth = depth.copy()
                depth[m == 0] = 0

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=depth_scale_o3d,
            depth_trunc=float(args.depth_trunc),
            convert_rgb_to_intensity=False
        )

        # node_poses[i] 是 camera->world (T_cw)，TSDF integrate 需要 world->camera
        T_wc = np.linalg.inv(T_cw)
        volume.integrate(rgbd, pin, T_wc)

        if (i + 1) % 5 == 0:
            print(f"[TSDF] integrated {i+1}/{len(poses)}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    mesh_path = out_dir / "tsdf_mesh.ply"
    pcd_path = out_dir / "tsdf_model.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    print("Saved:", mesh_path)
    print("Saved:", pcd_path)


if __name__ == "__main__":
    main()
