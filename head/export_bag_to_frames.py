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


def _make_even_indices(total: int, n: int):
    """生成长度为 n 的均匀索引，保证严格递增且不越界"""
    if n <= 1:
        return np.array([0], dtype=int)
    raw = np.round(np.linspace(0, total - 1, n)).astype(int)

    # 去重/单调修正
    out = []
    last = -1
    for v in raw:
        v = int(v)
        if v <= last:
            v = last + 1
        if v >= total:
            v = total - 1
        out.append(v)
        last = v

    # 如果修正后末尾顶到 total-1 导致数量不够，往前挤
    out = np.array(out, dtype=int)
    if len(np.unique(out)) < n:
        out = np.unique(out)
    # 兜底：补齐到 n
    while out.size < n:
        cand = out[-1] - 1
        if cand < 0:
            break
        out = np.sort(np.unique(np.append(out, cand)))
    # 再裁成 n（一般不会发生）
    if out.size > n:
        out = out[:n]
    return out


def count_valid_frames(bag_path: Path):
    """第一遍：统计有效帧数（color+depth 都存在）"""
    pipeline, profile, align, _ = _start_playback(bag_path)

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


def export_uniform_samples(
    bag_path: Path,
    out_dir: Path,
    target_step_deg: float = 18.95,
    turn_degrees: float = 360.0,
    num_samples: int = 0,
):
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    # 1) 统计有效帧
    total = count_valid_frames(bag_path)
    print(f"[INFO] valid frames in bag = {total}")

    # 2) 决定抽样数量
    if num_samples <= 0:
        # 360/18.95 ≈ 19
        num_samples = int(round(turn_degrees / target_step_deg))
        num_samples = max(2, num_samples)

    if num_samples > total:
        num_samples = total

    src_indices = _make_even_indices(total, num_samples)
    print(f"[INFO] num_samples = {num_samples}")
    print(f"[INFO] selected src frame indices (0-based) = {src_indices.tolist()}")

    # 3) 第二遍：只导出选中的帧
    pipeline, profile, align, intrinsics = _start_playback(bag_path)
    (out_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")

    selected_set = set(int(x) for x in src_indices.tolist())

    saved = 0
    frame_id = 0  # 仅对“有效帧”计数（color+depth都有时才+1）

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            # 这是第 frame_id 个有效帧
            if frame_id in selected_set:
                color_img = np.asanyarray(color.get_data())
                depth_img = np.asanyarray(depth.get_data())  # uint16

                # 兼容 rgb8 / bgr8
                fmt = color.get_profile().format()
                if fmt == rs.format.rgb8:
                    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                elif fmt == rs.format.bgr8:
                    color_bgr = color_img
                else:
                    # 兜底：按 RGB 处理
                    if color_img.ndim == 3 and color_img.shape[2] == 3:
                        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                    else:
                        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)

                cv2.imwrite(str(out_frames / f"{saved:04d}_color.png"), color_bgr)
                cv2.imwrite(str(out_frames / f"{saved:04d}_depth.png"), depth_img)

                saved += 1
                if saved % 5 == 0 or saved == num_samples:
                    print(f"Saved {saved}/{num_samples} ... (src_frame={frame_id})")

                if saved >= num_samples:
                    break

            frame_id += 1

    except RuntimeError as e:
        print("Reached end of bag or runtime error:", e)
    finally:
        pipeline.stop()

    print(f"Done. Total saved frames = {saved}")

    # 4) 写索引文件
    (out_dir / "selected_src_indices.txt").write_text(
        "\n".join(map(str, src_indices.tolist())),
        encoding="utf-8"
    )
    # 因为只保存了 num_samples 帧，所以 subset 就是 0..num_samples-1
    subset = list(range(saved))
    (out_dir / f"subset{saved}.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")

    # 为兼容你之前的流程，如果正好是19帧，也写 subset19.txt
    if saved == 19:
        (out_dir / "subset19.txt").write_text("\n".join(map(str, subset)), encoding="utf-8")

    print("Wrote selected_src_indices.txt")
    print(f"Wrote subset{saved}.txt" + (" and subset19.txt" if saved == 19 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)

    # 你给的目标：相邻约 18.95°（默认转一圈360° -> 约19帧）
    ap.add_argument("--target_step_deg", type=float, default=18.95)
    ap.add_argument("--turn_degrees", type=float, default=360.0)

    # 如果你想强制抽 N 帧（比如论文用107帧），就传这个；0 表示按角度算
    ap.add_argument("--num_samples", type=int, default=0)

    args = ap.parse_args()
    export_uniform_samples(
        bag_path=Path(args.bag),
        out_dir=Path(args.out),
        target_step_deg=args.target_step_deg,
        turn_degrees=args.turn_degrees,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
