# 分段选最优帧 + 深度滤波 + 质量日志
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs


def _start_playback(bag_path: Path):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(cfg)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    intrinsics = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "distortion_model": str(intr.model),
        "distortion_coeffs": list(intr.coeffs),
        "depth_scale": depth_scale,
        "align_depth_to_color": True,
        "bag_path": str(bag_path),
    }
    return pipeline, profile, align, intrinsics


def count_valid_frames(bag_path: Path):
    pipeline, _, align, _ = _start_playback(bag_path)
    total = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                continue
            total += 1
    except RuntimeError:
        pass
    finally:
        pipeline.stop()
    return total


def _to_bgr(color_frame):
    img = np.asanyarray(color_frame.get_data())
    fmt = color_frame.get_profile().format()
    if fmt == rs.format.rgb8:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == rs.format.bgr8:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def sharpness_score(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def depth_valid_ratio(depth_u16, depth_scale, z_min_m=0.25, z_max_m=1.2, roi=None):
    d = depth_u16.astype(np.float32) * float(depth_scale)
    if roi is not None:
        x0, y0, x1, y1 = roi
        d = d[y0:y1, x0:x1]
    valid = (d >= float(z_min_m)) & (d <= float(z_max_m))
    return float(valid.mean())


def export_samples(
    bag_path: Path,
    out_dir: Path,
    target_step_deg: float = 18.95,
    turn_degrees: float = 360.0,
    num_samples: int = 0,
    strategy: str = "segment_best",  # uniform | segment_best
    filter_depth: int = 1,
    use_hole_filling: int = 0,
    zmin_m: float = 0.25,
    zmax_m: float = 1.2,
    w_sharp: float = 1.0,
    w_depth: float = 1.5,
):
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    total = count_valid_frames(bag_path)
    print(f"[INFO] valid frames in bag = {total}")

    if num_samples <= 0:
        num_samples = int(round(float(turn_degrees) / float(target_step_deg)))
        num_samples = max(2, num_samples)
    num_samples = min(num_samples, total)
    print(f"[INFO] num_samples = {num_samples}, strategy={strategy}")

    pipeline, _, align, intr = _start_playback(bag_path)
    (out_dir / "intrinsics.json").write_text(json.dumps(intr, indent=2), encoding="utf-8")

    # RealSense depth filters (optional)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole = rs.hole_filling_filter()

    # mild defaults (hair-friendly: 不要太猛)
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    # segmentation boundaries for segment_best
    edges = np.linspace(0, total, num_samples + 1).astype(int)

    best = [None] * num_samples  # each: dict(color, depth, src_frame_id, score, sharp, vr)
    frame_id = 0

    h, w = intr["height"], intr["width"]
    # center ROI for depth validity (avoid bg influence)
    roi = (int(0.35 * w), int(0.30 * h), int(0.65 * w), int(0.70 * h))

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            if filter_depth:
                depth = spatial.process(depth)
                depth = temporal.process(depth)
                if use_hole_filling:
                    depth = hole.process(depth)

            seg = None
            if strategy == "segment_best":
                # find which segment this valid frame belongs to
                # seg s.t. edges[seg] <= frame_id < edges[seg+1]
                seg = int(np.searchsorted(edges, frame_id, side="right") - 1)
                if seg < 0:
                    seg = 0
                if seg >= num_samples:
                    seg = num_samples - 1

                bgr = _to_bgr(color)
                dep = np.asanyarray(depth.get_data()).copy()  # uint16
                sh = sharpness_score(bgr)
                vr = depth_valid_ratio(dep, intr["depth_scale"], zmin_m, zmax_m, roi=roi)
                score = float(w_sharp * sh + w_depth * (vr * 1000.0))  # scale vr for comparable magnitude

                cur = best[seg]
                if (cur is None) or (score > cur["score"]):
                    best[seg] = {
                        "color_bgr": bgr,
                        "depth_u16": dep,
                        "src_frame_id": int(frame_id),
                        "sharp": float(sh),
                        "valid_ratio": float(vr),
                        "score": float(score),
                        "seg": int(seg),
                    }

            else:
                # uniform: pick exact indices (old behavior)
                # build indices once:
                pass

            frame_id += 1

    except RuntimeError:
        pass
    finally:
        pipeline.stop()

    # If strategy == uniform, fall back to old exact-even indices
    if strategy == "uniform":
        # compute raw even indices among valid frames
        idxs = np.round(np.linspace(0, total - 1, num_samples)).astype(int)
        idxs = np.clip(idxs, 0, total - 1).tolist()
        selected = set(idxs)

        pipeline, _, align, intr2 = _start_playback(bag_path)
        (out_dir / "intrinsics.json").write_text(json.dumps(intr2, indent=2), encoding="utf-8")

        saved = 0
        frame_id = 0
        meta = []
        try:
            while True:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    continue

                if filter_depth:
                    depth = spatial.process(depth)
                    depth = temporal.process(depth)
                    if use_hole_filling:
                        depth = hole.process(depth)

                if frame_id in selected:
                    bgr = _to_bgr(color)
                    dep = np.asanyarray(depth.get_data()).copy()
                    sh = sharpness_score(bgr)
                    vr = depth_valid_ratio(dep, intr2["depth_scale"], zmin_m, zmax_m, roi=roi)

                    cv2.imwrite(str(out_frames / f"{saved:04d}_color.png"), bgr)
                    cv2.imwrite(str(out_frames / f"{saved:04d}_depth.png"), dep)

                    meta.append({
                        "saved_id": int(saved),
                        "src_frame_id": int(frame_id),
                        "sharp": float(sh),
                        "valid_ratio": float(vr),
                    })

                    saved += 1
                    if saved >= num_samples:
                        break

                frame_id += 1
        except RuntimeError:
            pass
        finally:
            pipeline.stop()

        # write logs
        (out_dir / "selected_src_indices.txt").write_text("\n".join(map(str, idxs)), encoding="utf-8")
        subset = list(range(saved))
        (out_dir / f"subset{saved}.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")
        if saved == 19:
            (out_dir / "subset19.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")

        (out_dir / "frames_meta.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in meta),
            encoding="utf-8"
        )

        print(f"[DONE] saved={saved}, wrote subset{saved}.txt")
        return

    # strategy == segment_best: write best frames per segment
    meta = []
    saved = 0
    selected_src = []
    for seg in range(num_samples):
        item = best[seg]
        if item is None:
            continue
        bgr = item["color_bgr"]
        dep = item["depth_u16"]
        cv2.imwrite(str(out_frames / f"{saved:04d}_color.png"), bgr)
        cv2.imwrite(str(out_frames / f"{saved:04d}_depth.png"), dep)

        meta.append({
            "saved_id": int(saved),
            "seg": int(seg),
            "src_frame_id": int(item["src_frame_id"]),
            "sharp": float(item["sharp"]),
            "valid_ratio": float(item["valid_ratio"]),
            "score": float(item["score"]),
        })
        selected_src.append(int(item["src_frame_id"]))
        saved += 1

    (out_dir / "selected_src_indices.txt").write_text("\n".join(map(str, selected_src)), encoding="utf-8")
    subset = list(range(saved))
    (out_dir / f"subset{saved}.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")
    if saved == 19:
        (out_dir / "subset19.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")

    (out_dir / "frames_meta.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in meta),
        encoding="utf-8"
    )
    print(f"[DONE] saved={saved}, wrote subset{saved}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--target_step_deg", type=float, default=18.95)
    ap.add_argument("--turn_degrees", type=float, default=360.0)
    ap.add_argument("--num_samples", type=int, default=0)

    ap.add_argument("--strategy", choices=["segment_best", "uniform"], default="segment_best")
    ap.add_argument("--filter_depth", type=int, default=1)
    ap.add_argument("--use_hole_filling", type=int, default=0)

    ap.add_argument("--zmin_m", type=float, default=0.25)
    ap.add_argument("--zmax_m", type=float, default=1.2)
    ap.add_argument("--w_sharp", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=1.5)

    args = ap.parse_args()
    export_samples(
        bag_path=Path(args.bag),
        out_dir=Path(args.out),
        target_step_deg=args.target_step_deg,
        turn_degrees=args.turn_degrees,
        num_samples=args.num_samples,
        strategy=args.strategy,
        filter_depth=args.filter_depth,
        use_hole_filling=args.use_hole_filling,
        zmin_m=args.zmin_m,
        zmax_m=args.zmax_m,
        w_sharp=args.w_sharp,
        w_depth=args.w_depth,
    )


if __name__ == "__main__":
    main()
