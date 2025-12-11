# reconstruct_mesh.py
import open3d as o3d
import numpy as np

MERGED_INPUT = "merged_head.ply"
MESH_OUTPUT = "head_mesh.ply"


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(MERGED_INPUT)
    print(f"[INFO] Loaded merged point cloud, points={len(pcd.points)}")

    # 法向估计
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=50
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    print("[INFO] Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    print(f"[INFO] Mesh vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")

    # 基于密度裁剪（去掉特别稀疏的区域）
    densities = np.asarray(densities)
    density_thresh = np.quantile(densities, 0.05)  # 去掉最稀疏 5%
    vertices_to_remove = densities < density_thresh
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(MESH_OUTPUT, mesh)
    print(f"[DONE] Saved mesh to {MESH_OUTPUT}")
