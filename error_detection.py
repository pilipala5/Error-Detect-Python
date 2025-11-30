from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


@dataclass
class InnerParams:
    """相机内方位元素。"""

    f: float
    x0: float
    y0: float


@dataclass
class ImageData:
    """单张影像的像点观测。"""

    image_id: int
    points: Dict[int, Tuple[float, float]]


@dataclass
class ErrorPoint:
    """粗差点信息，误差值单位为σ。"""

    point_id: int
    error_sigma: float
    error_abs_sigma: float


def read_inner_params(ioe_path: Path) -> InnerParams:
    """
    读取 PRECAI.DAT，格式与原 C++ 相同：
    第 1 行为无用的提示行，第 2 行开始为 f x0 y0。
    """
    with ioe_path.open("r", encoding="utf-8") as f:
        f.readline()  # 丢弃首行提示
        parts = f.readline().split()
    if len(parts) < 3:
        raise ValueError(f"{ioe_path} 中缺少内方位元素")
    return InnerParams(f=float(parts[0]), x0=float(parts[1]), y0=float(parts[2]))


def read_pixel_file(pixel_path: Path) -> List[ImageData]:
    """
    读取像点文件（PREPHIy10.DAT 等）。
    文件中包含多张影像的观测，结构为：
        1) 首行提示，直接跳过
        2) 每块数据首行是影像 ID（可能为负号，表示后续取绝对值）
        3) 之后若干行为 point_id x y，当 point_id == -99 时结束当前影像
    返回按文件出现顺序排列的影像列表。
    """
    images: List[ImageData] = []
    with pixel_path.open("r", encoding="utf-8") as f:
        # 跳过首行提示
        f.readline()

        while True:
            id_line = f.readline()
            if not id_line:
                break
            id_line = id_line.strip()
            if not id_line:
                continue

            image_id = abs(int(float(id_line.split()[0])))
            points: Dict[int, Tuple[float, float]] = {}

            for line in f:
                parts = line.split()
                if not parts:
                    continue
                pid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                if pid == -99:
                    break
                points[pid] = (x, y)

            if points:
                images.append(ImageData(image_id=image_id, points=points))

    return images


class LpvsDetector:
    """
    仅保留 METHOD_ID=1（Lpvs）的粗差检测器。
    按原 C++ 思路：相对定向 + 逐步选权，P(i,i) < 0.001 记为粗差候选。
    """

    def __init__(self, inner: InnerParams, m0: float = 2.8e-3) -> None:
        self.inner = inner
        self.m0 = m0  # 单位 mm，对应 C++ 中的 m0

    @staticmethod
    def _orientation(phi: float, omega: float, kappa: float) -> Tuple[float, ...]:
        """计算旋转矩阵参数，便于复用。"""
        a1 = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
        a2 = -math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
        a3 = -math.sin(phi) * math.cos(omega)
        b1 = math.cos(omega) * math.sin(kappa)
        b2 = math.cos(omega) * math.cos(kappa)
        b3 = -math.sin(omega)
        c1 = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
        c2 = -math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
        c3 = math.cos(phi) * math.cos(omega)
        return a1, a2, a3, b1, b2, b3, c1, c2, c3

    def detect_pair(self, img_a: ImageData, img_b: ImageData) -> Tuple[List[ErrorPoint], dict]:
        """
        对一对影像执行 Lpvs 粗差检测。
        返回满足 4-100σ 范围的粗差列表及相关统计。
        """
        common_ids = [pid for pid in img_a.points if pid in img_b.points]
        if len(common_ids) < 6:
            raise ValueError(f"影像对 {img_a.image_id}-{img_b.image_id} 公共点过少，无法解算")

        num = len(common_ids)
        P = np.eye(num)
        A = np.zeros((num, 5))
        Q = np.zeros((num, 1))
        e_points: Set[int] = set()

        phi = omega = kappa = u = v = 0.0
        loop1 = 0
        last_AtPA = None  # 持有最新的正规方程，供外层迭代使用

        while True:
            loop1 += 1
            loop2 = 0
            while True:
                loop2 += 1
                Bx = 1.0
                By = Bx * u
                Bz = Bx * v

                a1, a2, a3, b1, b2, b3, c1, c2, c3 = self._orientation(phi, omega, kappa)

                # 逐点构建 A、Q
                for i, pid in enumerate(common_ids):
                    X1, Y1 = img_a.points[pid]
                    Z1 = -self.inner.f
                    x2, y2 = img_b.points[pid]
                    X2 = a1 * x2 + a2 * y2 - a3 * self.inner.f
                    Y2 = b1 * x2 + b2 * y2 - b3 * self.inner.f
                    Z2 = c1 * x2 + c2 * y2 - c3 * self.inner.f

                    denom = X1 * Z2 - Z1 * X2
                    N1 = (Bx * Z2 - Bz * X2) / denom
                    N2 = (Bx * Z1 - Bz * X1) / denom

                    A[i, 0] = -X2 * Y2 * N2 / Z2
                    A[i, 1] = -(Z2 + Y2 * Y2 / Z2) * N2
                    A[i, 2] = X2 * N2
                    A[i, 3] = Bx
                    A[i, 4] = -Y2 * Bx / Z2

                    Q[i, 0] = N1 * Y1 - N2 * Y2 - By

                AtPA = A.T @ P @ A
                last_AtPA = AtPA
                DX = np.linalg.solve(AtPA, A.T @ P @ Q)

                phi += DX[0, 0]
                omega += DX[1, 0]
                kappa += DX[2, 0]
                u += DX[3, 0]
                v += DX[4, 0]

                if float(np.max(np.abs(DX))) < 3e-5 or loop2 >= 30:
                    break

            # 计算残差、单位权中误差等
            V = A @ DX - Q
            AtPA_inv = np.linalg.inv(last_AtPA)
            Qvv = np.linalg.inv(P) - A @ AtPA_inv @ A.T
            sigma0 = math.sqrt(float((V.T @ P @ V)[0, 0] / (num - 5)))
            d_val = 3.5 + 82 / (81 + (sigma0 / 0.0028) ** 4)

            # 依据 Lpvs 重新选权
            P_prev = P.copy()
            for i, pid in enumerate(common_ids):
                Ti = float(V[i, 0] ** 2 / (sigma0 * sigma0 * Qvv[i, i] * P[i, i]))
                K = 1.0 if loop1 <= 3 else 3.29
                if Ti <= K:
                    P[i, i] = 1.0
                else:
                    P[i, i] = 1.0 / Ti
                if P[i, i] < 0.001:
                    e_points.add(pid)

            if float(np.max(np.abs(P_prev - P))) < 0.001 or loop1 >= 30:
                break

        # 最终姿态下计算粗差并过滤 4-100σ
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = self._orientation(phi, omega, kappa)
        Bx = 1.0
        By = Bx * u
        Bz = Bx * v
        errors: List[ErrorPoint] = []
        for pid in e_points:
            X1, Y1 = img_a.points[pid]
            Z1 = -self.inner.f
            x2, y2 = img_b.points[pid]
            X2 = a1 * x2 + a2 * y2 - a3 * self.inner.f
            Y2 = b1 * x2 + b2 * y2 - b3 * self.inner.f
            Z2 = c1 * x2 + c2 * y2 - c3 * self.inner.f

            denom = X1 * Z2 - Z1 * X2
            N1 = (Bx * Z2 - Bz * X2) / denom
            N2 = (Bx * Z1 - Bz * X1) / denom

            dy = (N1 * Y1 - By) / N2 - Y2
            err_sigma = dy / self.m0
            err_abs = abs(err_sigma)

            if 4 <= err_abs <= 100:
                errors.append(ErrorPoint(point_id=pid, error_sigma=err_sigma, error_abs_sigma=err_abs))

        errors.sort(key=lambda e: e.point_id)

        stats = {
            "pair": f"{img_a.image_id}-{img_b.image_id}",
            "total_points": num,
            "candidate_errors": len(e_points),
            "filtered_errors": len(errors),
        }
        return errors, stats


def fmt_errors(label: str, errors: Iterable[ErrorPoint]) -> List[str]:
    """将粗差结果格式化为文本行，便于统一输出。"""
    lines = [f"影像对 {label} 粗差点（4-100σ）："]
    lines.append("PointID\t误差(σ)\t绝对值(σ)")
    for ep in errors:
        lines.append(f"{ep.point_id}\t{ep.error_sigma:.6f}\t{ep.error_abs_sigma:.6f}")
    if len(lines) == 2:
        lines.append("无粗差点满足筛选条件。")
    return lines
