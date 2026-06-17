#!/usr/bin/env python3
"""
msh_to_pyvista_views.py

Load a Gmsh .msh file into PyVista and generate four standard scientific views:
1) surface rendering + edges
2) clipped interior view
3) orthogonal slice view
4) cell-quality view

Examples
--------
python msh_to_pyvista_views.py my_mesh.msh
python msh_to_pyvista_views.py my_mesh.msh -o figures
python msh_to_pyvista_views.py my_mesh.msh --quality-measure scaled_jacobian
python msh_to_pyvista_views.py my_mesh.msh --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate four PyVista views from a Gmsh .msh file."
    )
    parser.add_argument("mesh_file", type=Path, help="Path to input .msh file")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <mesh_stem>_pyvista_views)",
    )
    parser.add_argument(
        "--quality-measure",
        default="scaled_jacobian",
        help="PyVista cell quality measure to use (default: scaled_jacobian)",
    )
    parser.add_argument(
        "--clip-normal",
        default="-x",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        help="Plane normal for clipped views (default: -x)",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=[1800, 1400],
        metavar=("W", "H"),
        help="Render window size in pixels (default: 1800 1400)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive windows instead of only saving PNGs",
    )
    return parser.parse_args()


def read_mesh(path: Path) -> pv.DataSet:
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = pv.read(path)

    # If a MultiBlock is returned, combine non-empty blocks into one dataset.
    if isinstance(mesh, pv.MultiBlock):
        blocks = [b for b in mesh if b is not None and getattr(b, "n_cells", 0) > 0]
        if not blocks:
            raise ValueError("The .msh file was read, but no non-empty blocks were found.")
        mesh = pv.MultiBlock(blocks).combine(merge_points=True)

    if not isinstance(mesh, pv.DataSet):
        raise TypeError(f"Unsupported dataset type returned by PyVista: {type(mesh)}")

    if mesh.n_cells == 0:
        raise ValueError("Mesh contains zero cells.")

    return mesh


def extract_surface_safe(mesh: pv.DataSet) -> pv.PolyData:
    try:
        surf = mesh.extract_surface()
        if surf.n_cells > 0:
            return surf
    except Exception:
        pass

    # Fallback for already-surface-like datasets
    if isinstance(mesh, pv.PolyData):
        return mesh
    raise RuntimeError("Could not extract a valid surface from the mesh.")


def compute_quality(mesh: pv.DataSet, measure: str):
    """
    Try the modern cell_quality() first.
    Fall back to compute_cell_quality() for older PyVista versions.
    Returns: (quality_mesh, array_name)
    """
    try:
        qmesh = mesh.cell_quality(quality_measure=measure, null_value=np.nan)
        return qmesh, measure
    except Exception:
        # Older API fallback
        qmesh = mesh.compute_cell_quality(quality_measure=measure)
        array_name = "CellQuality"
        arr = np.asarray(qmesh.cell_data[array_name], dtype=float)
        arr[arr < 0] = np.nan
        qmesh.cell_data[array_name] = arr
        return qmesh, array_name


def add_common_scene_items(pl: pv.Plotter, mesh: pv.DataSet):
    pl.add_axes()
    pl.show_grid(color="lightgray")
    pl.add_mesh(mesh.outline(), color="black", line_width=1)
    pl.background_color = "white"


def save_or_show(plotter: pv.Plotter, png_path: Path | None, interactive: bool):
    if interactive:
        plotter.show()
    else:
        if png_path is None:
            raise ValueError("png_path must be provided when interactive=False")
        plotter.show(screenshot=str(png_path), auto_close=True)


def main():
    args = parse_args()

    mesh = read_mesh(args.mesh_file)
    surface = extract_surface_safe(mesh)

    outdir = args.outdir or (args.mesh_file.parent / f"{args.mesh_file.stem}_pyvista_views")
    outdir.mkdir(parents=True, exist_ok=True)

    center = mesh.center
    window_size = tuple(args.window_size)

    print(f"Loaded mesh: {args.mesh_file}")
    print(f"  points: {mesh.n_points}")
    print(f"  cells:  {mesh.n_cells}")
    print(f"  bounds: {mesh.bounds}")
    print(f"  center: {center}")
    print(f"Saving outputs to: {outdir}")

    # ------------------------------------------------------------------
    # 1) Surface rendering + edges
    # ------------------------------------------------------------------
    pl = pv.Plotter(off_screen=not args.interactive, window_size=window_size)
    add_common_scene_items(pl, mesh)
    pl.add_text("1) Surface rendering + edges", font_size=16, color="black")
    pl.add_mesh(
        surface,
        color="lightsteelblue",
        show_edges=True,
        edge_color="black",
        line_width=0.8,
        smooth_shading=False,
        opacity=1.0,
    )
    pl.camera_position = "iso"
    save_or_show(
        pl,
        outdir / "01_surface_edges.png",
        interactive=args.interactive,
    )

    # ------------------------------------------------------------------
    # 2) Clipped interior view
    # ------------------------------------------------------------------
    clipped = mesh.clip(normal=args.clip_normal, origin=center, invert=False)

    pl = pv.Plotter(off_screen=not args.interactive, window_size=window_size)
    add_common_scene_items(pl, mesh)
    pl.add_text("2) Clipped interior view", font_size=16, color="black")

    # translucent context
    pl.add_mesh(
        surface,
        color="lightgray",
        opacity=0.18,
        show_edges=False,
    )
    # clipped mesh
    pl.add_mesh(
        clipped,
        color="lightskyblue",
        show_edges=True,
        edge_color="black",
        line_width=0.6,
        smooth_shading=False,
    )
    pl.camera_position = "iso"
    save_or_show(
        pl,
        outdir / "02_clipped_view.png",
        interactive=args.interactive,
    )

    # ------------------------------------------------------------------
    # 4) Cell-quality view
    # ------------------------------------------------------------------
    qmesh, qname = compute_quality(mesh, args.quality_measure)
    qclip = qmesh.clip(normal=args.clip_normal, origin=center, invert=False)

    arr = np.asarray(qclip.cell_data[qname], dtype=float)
    finite = np.isfinite(arr)

    pl = pv.Plotter(off_screen=not args.interactive, window_size=window_size)
    add_common_scene_items(pl, mesh)
    pl.add_text(f"3) Cell quality: {args.quality_measure}", font_size=16, color="black")

    # translucent shell for context
    pl.add_mesh(
        surface,
        color="lightgray",
        opacity=0.10,
        show_edges=False,
    )

    if np.any(finite):
        clim = [float(np.nanmin(arr)), float(np.nanmax(arr))]
        pl.add_mesh(
            qclip,
            scalars=qname,
            cmap="viridis",
            clim=clim,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            scalar_bar_args={"title": args.quality_measure},
            nan_color="red",
        )
    else:
        print(
            f"Warning: quality measure '{args.quality_measure}' produced no finite values "
            f"for this mesh. Plotting clipped mesh without scalars."
        )
        pl.add_mesh(
            qclip,
            color="lightcoral",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
        )

    pl.camera_position = "iso"
    save_or_show(
        pl,
        outdir / "03_cell_quality.png",
        interactive=args.interactive,
    )

    print("Done.")
    print("Generated:")
    print(f"  {outdir / '01_surface_edges.png'}")
    print(f"  {outdir / '02_clipped_view.png'}")
    print(f"  {outdir / '03_orthogonal_slices.png'}")
    print(f"  {outdir / '04_cell_quality.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
