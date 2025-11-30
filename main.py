from __future__ import annotations

"""
基于 Lpvs 的粗差检测主程序。
读取 data 目录下的观测文件，计算 1-2、2-3 两个影像对的粗差点，并输出单对结果与两对共同粗差点。
"""

from pathlib import Path

from error_detection import (
    LpvsDetector,
    ErrorPoint,
    fmt_errors,
    read_inner_params,
    read_pixel_file,
)


def build_pairs(images):
    """
    按文件出现顺序组装影像对：
    - 第 1 与第 2 张影像视作“1-2 图”
    - 第 2 与第 3 张影像视作“2-3 图”
    若影像数量不足，将返回空列表。
    """
    pairs = []
    if len(images) >= 2:
        pairs.append(("1-2", images[0], images[1]))
    if len(images) >= 3:
        pairs.append(("2-3", images[1], images[2]))
    return pairs


def write_output(out_path: Path, pair_lines, common_lines, stats_lines):
    """统一写出到文本文件，便于查阅。"""
    content = []
    content.append("== 粗差检测统计 ==")
    content.extend(stats_lines)
    content.append("")
    content.extend(pair_lines)
    content.append("== 两对共同粗差点 ==")
    content.extend(common_lines)
    out_path.write_text("\n".join(content), encoding="utf-8")


def main():
    data_dir = Path(__file__).parent / "data"
    ioe_path = data_dir / "PRECAI.DAT"
    pixel_path = data_dir / "PREPHIy10.DAT"

    # 读取内方位与像点
    inner = read_inner_params(ioe_path)
    images = read_pixel_file(pixel_path)

    pairs = build_pairs(images)
    if not pairs:
        raise SystemExit("影像数量不足，至少需要两张影像才能检测。")

    detector = LpvsDetector(inner=inner)
    pair_lines = []
    stats_lines = []
    pair_errors: dict[str, list[ErrorPoint]] = {}

    for label, img_a, img_b in pairs:
        errors, stats = detector.detect_pair(img_a, img_b)
        pair_label = f"{label}（ID {stats['pair']}）"
        pair_errors[pair_label] = errors
        stats_lines.append(
            f"{pair_label}: 公共点 {stats['total_points']}，候选粗差 {stats['candidate_errors']}，筛选后(4-100σ) {stats['filtered_errors']}"
        )

        pair_lines.extend(fmt_errors(pair_label, errors))
        pair_lines.append("")  # 分隔空行
        # 这里不再输出合并列表

    # 计算两对影像的共同粗差点
    common_lines = []
    common_count = 0
    if len(pair_errors) >= 2:
        labels = list(pair_errors.keys())
        pid_maps = {lbl: {ep.point_id: ep for ep in eps} for lbl, eps in pair_errors.items()}
        common_ids = set.intersection(*(set(m.keys()) for m in pid_maps.values()))
        common_count = len(common_ids)

        if common_ids:
            header = ["PointID"]
            for lbl in labels:
                header.append(f"{lbl}粗差(mm)")
                header.append(f"{lbl}倍数")
            common_lines.append("\t".join(header))

            for pid in sorted(common_ids):
                row = [str(pid)]
                for lbl in labels:
                    ep = pid_maps[lbl][pid]
                    row.append(f"{ep.error_mm:.6f}")
                    row.append(f"{ep.rel_abs_sigma:.6f}")
                common_lines.append("\t".join(row))
        else:
            common_lines.append("无两对共同的粗差点。")
    else:
        common_lines.append("仅检测到一对影像，无法统计共同粗差点。")

    # 将共同粗差点个数加入统计
    stats_lines.append(f"两对共同粗差点: {common_count}")

    out_path = data_dir / "lpvs_result.txt"
    write_output(out_path, pair_lines, common_lines, stats_lines)
    print(f"粗差检测完成，结果已写入 {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
