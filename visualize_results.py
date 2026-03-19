from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from packing_kernel import canonicalize_placement, normalize_items
from verifier import BoxVerifier


NON_FRAGILE_COLOR = "#4f8bd6"
FRAGILE_COLOR = "#d65a4f"
CONTAINER_COLOR = "#444444"
ORIGIN_COLOR = "#111111"


def cube_faces(origin: Tuple[int, int, int], size: Tuple[int, int, int]):
    x, y, z = origin
    dx, dy, dz = size
    return [
        [[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z]],
        [[x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]],
        [[x, y, z], [x, y + dy, z], [x, y + dy, z + dz], [x, y, z + dz]],
        [[x + dx, y, z], [x + dx, y + dy, z], [x + dx, y + dy, z + dz], [x + dx, y, z + dz]],
        [[x, y, z], [x + dx, y, z], [x + dx, y, z + dz], [x, y, z + dz]],
        [[x, y + dy, z], [x + dx, y + dy, z], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]],
    ]


def draw_container_wireframe(ax, bin_size: Sequence[int]) -> None:
    L, W, H = bin_size
    edges = [
        ((0, 0, 0), (L, 0, 0)),
        ((0, 0, 0), (0, W, 0)),
        ((0, 0, 0), (0, 0, H)),
        ((L, W, 0), (0, W, 0)),
        ((L, W, 0), (L, 0, 0)),
        ((L, W, 0), (L, W, H)),
        ((L, 0, H), (0, 0, H)),
        ((L, 0, H), (L, 0, 0)),
        ((L, 0, H), (L, W, H)),
        ((0, W, H), (0, 0, H)),
        ((0, W, H), (0, W, 0)),
        ((0, W, H), (L, W, H)),
    ]
    for (x1, y1, z1), (x2, y2, z2) in edges:
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=CONTAINER_COLOR, linewidth=1.0)


def _annotate_rect(ax, x: int, y: int, w: int, h: int, label: str) -> None:
    ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=8)


def _configure_2d_axes(ax, width: int, height: int, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=CONTAINER_COLOR, linewidth=1.2))


def _draw_origin_marker_2d(ax, width: int, height: int, xlabel: str, ylabel: str) -> None:
    arrow_x = max(40, int(round(width * 0.08)))
    arrow_y = max(40, int(round(height * 0.08)))
    ax.scatter([0], [0], s=26, color=ORIGIN_COLOR, zorder=5)
    ax.text(10, 10, "O", color=ORIGIN_COLOR, fontsize=9, ha="left", va="bottom")
    ax.add_patch(
        FancyArrowPatch((0, 0), (arrow_x, 0), arrowstyle="->", mutation_scale=10, linewidth=1.2, color=ORIGIN_COLOR)
    )
    ax.add_patch(
        FancyArrowPatch((0, 0), (0, arrow_y), arrowstyle="->", mutation_scale=10, linewidth=1.2, color=ORIGIN_COLOR)
    )
    ax.text(arrow_x, 0, f"  +{xlabel}", color=ORIGIN_COLOR, fontsize=8, va="bottom")
    ax.text(0, arrow_y, f"+{ylabel}", color=ORIGIN_COLOR, fontsize=8, ha="left", va="bottom")


def _draw_origin_marker_side_looking_pos_x(ax, width: int, height: int) -> None:
    arrow_y = max(40, int(round(width * 0.08)))
    arrow_z = max(40, int(round(height * 0.08)))
    ax.scatter([0], [0], s=26, color=ORIGIN_COLOR, zorder=5)
    ax.text(12, 10, "O", color=ORIGIN_COLOR, fontsize=9, ha="left", va="bottom")
    ax.add_patch(
        FancyArrowPatch((0, 0), (arrow_y, 0), arrowstyle="->", mutation_scale=10, linewidth=1.2, color=ORIGIN_COLOR)
    )
    ax.add_patch(
        FancyArrowPatch((0, 0), (0, arrow_z), arrowstyle="->", mutation_scale=10, linewidth=1.2, color=ORIGIN_COLOR)
    )
    ax.text(arrow_y, 0, "  +Y", color=ORIGIN_COLOR, fontsize=8, va="bottom")
    ax.text(0, arrow_z, "+Z", color=ORIGIN_COLOR, fontsize=8, ha="left", va="bottom")


def _configure_3d_axes(ax, bin_size: Sequence[int], legend_handles: List[Rectangle]) -> None:
    L, W, H = bin_size
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((L, W, H))
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")

    ax.set_xlabel("X", labelpad=8)
    ax.set_ylabel("Y", labelpad=8)
    ax.set_zlabel("Z", labelpad=6)
    ax.set_title("3D view")
    ax.view_init(elev=22, azim=-60)

    ax.set_xticks([0, L // 2, L])
    ax.set_yticks([0, W // 2, W])
    ax.set_zticks([0, H // 2, H])

    draw_container_wireframe(ax, bin_size)

    guide_x = max(L * 0.10, 60)
    guide_y = max(W * 0.10, 60)
    guide_z = max(H * 0.10, 60)
    ax.scatter([0], [0], [0], s=28, color=ORIGIN_COLOR, depthshade=False)
    ax.text(0, 0, 0, "  O(0,0,0)", fontsize=9, color=ORIGIN_COLOR)
    ax.plot([0, guide_x], [0, 0], [0, 0], color=ORIGIN_COLOR, linewidth=2.0)
    ax.plot([0, 0], [0, guide_y], [0, 0], color=ORIGIN_COLOR, linewidth=2.0)
    ax.plot([0, 0], [0, 0], [0, guide_z], color=ORIGIN_COLOR, linewidth=2.0)
    ax.text(guide_x, 0, 0, " +X", fontsize=8, color=ORIGIN_COLOR)
    ax.text(0, guide_y, 0, " +Y", fontsize=8, color=ORIGIN_COLOR)
    ax.text(0, 0, guide_z, " +Z", fontsize=8, color=ORIGIN_COLOR)

    try:
        ax.xaxis.pane.set_alpha(0.04)
        ax.yaxis.pane.set_alpha(0.04)
        ax.zaxis.pane.set_alpha(0.04)
    except Exception:
        pass
    ax.grid(True, alpha=0.18)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left")


def render_validated_solution(solution: Dict[str, Any], instance_name: str, save_path: str) -> None:
    items = normalize_items(solution["items"])
    placement = canonicalize_placement(solution["placement"])
    bin_size = tuple(solution["bin_size"])

    verifier = BoxVerifier(bin_size)
    ok, msg, _report = verifier.verify(items, placement, return_report=True)
    if not ok:
        raise ValueError(f"Refusing to visualize invalid solution: {msg}")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_front = fig.add_subplot(gs[1, 0])
    ax_side = fig.add_subplot(gs[1, 1])

    legend_handles: List[Rectangle] = []
    added_labels = set()
    for p in placement:
        item = items[p["id"]]
        color = FRAGILE_COLOR if item["fragile"] else NON_FRAGILE_COLOR
        label = "Fragile" if item["fragile"] else "Non-fragile"
        faces = cube_faces((p["x"], p["y"], p["z"]), (item["l"], item["w"], item["h"]))
        poly = Poly3DCollection(faces, facecolors=color, edgecolors="black", linewidths=0.7, alpha=0.72)
        ax3d.add_collection3d(poly)
        ax3d.text(
            p["x"] + item["l"] / 2.0,
            p["y"] + item["w"] / 2.0,
            p["z"] + item["h"] / 2.0,
            str(p["id"]),
            fontsize=8,
            ha="center",
            va="center",
        )
        if label not in added_labels:
            added_labels.add(label)
            legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", label=label))

    for p in sorted(placement, key=lambda pos: (pos["z"], pos["id"])):
        item = items[p["id"]]
        color = FRAGILE_COLOR if item["fragile"] else NON_FRAGILE_COLOR
        rect_top = Rectangle((p["x"], p["y"]), item["l"], item["w"], facecolor=color, edgecolor="black", alpha=0.78)
        ax_top.add_patch(rect_top)
        _annotate_rect(ax_top, p["x"], p["y"], item["l"], item["w"], str(p["id"]))

    for p in sorted(placement, key=lambda pos: (-pos["y"], pos["z"], pos["id"])):
        item = items[p["id"]]
        color = FRAGILE_COLOR if item["fragile"] else NON_FRAGILE_COLOR
        rect_front = Rectangle((p["x"], p["z"]), item["l"], item["h"], facecolor=color, edgecolor="black", alpha=0.78)
        ax_front.add_patch(rect_front)
        _annotate_rect(ax_front, p["x"], p["z"], item["l"], item["h"], str(p["id"]))

    for p in sorted(placement, key=lambda pos: (-pos["x"], pos["z"], pos["id"])):
        item = items[p["id"]]
        color = FRAGILE_COLOR if item["fragile"] else NON_FRAGILE_COLOR
        rect_side = Rectangle((p["y"], p["z"]), item["w"], item["h"], facecolor=color, edgecolor="black", alpha=0.78)
        ax_side.add_patch(rect_side)
        _annotate_rect(ax_side, p["y"], p["z"], item["w"], item["h"], str(p["id"]))

    _configure_3d_axes(ax3d, bin_size, legend_handles)
    _configure_2d_axes(ax_top, bin_size[0], bin_size[1], "Top view (X-Y)", "X", "Y")
    _configure_2d_axes(ax_front, bin_size[0], bin_size[2], "Front view (X-Z, looking along +Y)", "X", "Z")
    _configure_2d_axes(ax_side, bin_size[1], bin_size[2], "Side view (Y-Z, looking along +X)", "Y", "Z")
    ax_side.invert_xaxis()

    _draw_origin_marker_2d(ax_top, bin_size[0], bin_size[1], "X", "Y")
    _draw_origin_marker_2d(ax_front, bin_size[0], bin_size[2], "X", "Z")
    _draw_origin_marker_side_looking_pos_x(ax_side, bin_size[1], bin_size[2])

    title = f"3D Bin Packing - {instance_name}\nVerified: {msg} | Items: {len(items)} | Bin: {tuple(bin_size)}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_instance(results_folder: str, instance_name: str, save_path: str | None = None) -> bool:
    solution_file = os.path.join(results_folder, instance_name, "solution.json")
    if not os.path.exists(solution_file):
        print(f"Solution file not found: {solution_file}")
        return False
    with open(solution_file, "r", encoding="utf-8") as f:
        solution = json.load(f)

    if save_path is None:
        save_path = os.path.join("visualizations", f"{instance_name}.png")

    try:
        render_validated_solution(solution, instance_name, save_path)
    except Exception as exc:
        print(f"Visualization failed for {instance_name}: {exc}")
        return False

    print(f"Saved visualization: {save_path}")
    return True


def visualize_all_instances(results_folder: str, output_folder: str = "visualizations") -> None:
    os.makedirs(output_folder, exist_ok=True)
    instances = sorted(name for name in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, name)))
    success_count = 0
    for idx, instance in enumerate(instances, start=1):
        print(f"[{idx}/{len(instances)}] Visualizing {instance} ...")
        save_path = os.path.join(output_folder, f"{instance}.png")
        if visualize_instance(results_folder, instance, save_path):
            success_count += 1
    print(f"Visualization complete: {success_count}/{len(instances)} successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default=None, help="Visualize a single results/<instance> folder")
    parser.add_argument("--all", action="store_true", help="Visualize every instance under results/")
    parser.add_argument("--results", type=str, default="./results", help="Results folder path")
    parser.add_argument("--output", type=str, default="./visualizations", help="Output folder")
    args = parser.parse_args()

    if args.instance:
        visualize_instance(args.results, args.instance, os.path.join(args.output, f"{args.instance}.png"))
    elif args.all:
        visualize_all_instances(args.results, args.output)
    else:
        print("Usage:")
        print("  python visualize_results.py --instance <id>")
        print("  python visualize_results.py --all")
