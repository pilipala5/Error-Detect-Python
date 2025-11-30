from __future__ import annotations

"""
基于 Lpvs 的粗差检测主程序。
读取 data 目录下的观测文件，计算 1-2、2-3 两个影像对的粗差点并输出合并结果。
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


def write_output(out_path: Path, pair_lines, overall_lines, stats_lines):
    """统一写出到文本文件，便于查阅。"""
    content = []
    content.append("== 粗差检测统计 ==")
    content.extend(stats_lines)
    content.append("")
    content.extend(pair_lines)
    content.append("== 1-2 与 2-3 合并列表 ==")
    content.extend(overall_lines)
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
    overall: list[tuple[str, ErrorPoint]] = []
    stats_lines = []

    for label, img_a, img_b in pairs:
        errors, stats = detector.detect_pair(img_a, img_b)
        pair_label = f"{label}（ID {stats['pair']}）"
        stats_lines.append(
            f"{pair_label}: 公共点 {stats['total_points']}，候选粗差 {stats['candidate_errors']}，筛选后(4-100σ) {stats['filtered_errors']}"
        )

        pair_lines.extend(fmt_errors(pair_label, errors))
        pair_lines.append("")  # 分隔空行

        for ep in errors:
            overall.append((pair_label, ep))

    # 汇总两个影像对的粗差点
    overall_lines = []
    if overall:
        overall.sort(key=lambda item: (item[0], item[1].point_id))
        overall_lines.append("影像对\tPointID\t误差(σ)\t绝对值(σ)")
        for label, ep in overall:
            overall_lines.append(f"{label}\t{ep.point_id}\t{ep.error_sigma:.6f}\t{ep.error_abs_sigma:.6f}")
    else:
        overall_lines.append("无粗差点满足 4-100σ 筛选条件。")

    out_path = data_dir / "lpvs_result.txt"
    write_output(out_path, pair_lines, overall_lines, stats_lines)
    print(f"粗差检测完成，结果已写入 {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
