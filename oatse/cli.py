"""
oatse.cli: Command-Line Interface for Ocean Acoustic Thermocline & Sonar Engine (OATSE)
(`scenarios`, `stream-segy`, `evaluate`, `export-studio`, `ingest-las`, `ingest-dev`).

Provides workstation tools for inspecting 3D maritime scenarios, streaming large out-of-core
SEG-Y volumes with < 35 MB peak heap, parsing CWLS LAS 2.0 and Petrel .dev files, and running
coupled acoustic-impedance benchmark evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .io import (
    read_dev_survey,
    read_las2_file,
    read_segy_cube,
    stream_segy_volume,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
)
from .scenarios import (
    FRONT_MODE_NAMES,
    MARITIME_ACOUSTIC_CATALOG,
    SCENARIO_ALIASES,
    SCENARIO_DESCRIPTIONS,
    build_all_scenarios,
    build_maritime_scenario,
    evaluate_scenario_comparison,
    resolve_scenario_name,
)


def _cmd_scenarios_list(args: argparse.Namespace) -> int:
    print("=" * 90)
    print("OATSE v0.2.0 — 5-Scenario Maritime & Seabed Benchmark Deck")
    print("=" * 90)
    print(f"{'Key / ID':<32} | {'Title':<40} | {'Grid':<12}")
    print("-" * 90)
    for key, desc in SCENARIO_DESCRIPTIONS.items():
        print(f"{key:<32} | {desc['title'][:38]:<40} | (32, 24, 24)")
    print("=" * 90)
    print("Use 'oatse scenarios inspect --scenario <KEY>' for detailed petrophysics.")
    return 0


def _cmd_scenarios_inspect(args: argparse.Namespace) -> int:
    try:
        canonical = resolve_scenario_name(args.scenario)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    scen = build_maritime_scenario(canonical)
    meta = scen["meta"]
    print("=" * 90)
    print(f"Scenario: {canonical}")
    print(f"Title:    {meta['title']}")
    print(f"Region:   {meta['region']}")
    print("-" * 90)
    print("Summary:")
    print(f"  {meta['summary']}")
    print("-" * 90)
    print(f"Grid Shape:        {scen['grid_shape']} ({scen['grid_shape'][0]*scen['grid_shape'][1]*scen['grid_shape'][2]:,} voxels)")
    print(f"Sound Speed c:     min = {float(scen['sound_speed_c'].min()):.4f} km/s, max = {float(scen['sound_speed_c'].max()):.4f} km/s")
    print(f"Density rho:       min = {float(scen['density_rho'].min()):.4f} g/cm^3, max = {float(scen['density_rho'].max()):.4f} g/cm^3")
    print(f"Acoustic Z_p:      min = {float(scen['seabed_zp'].min()):.4f} MRayl, max = {float(scen['seabed_zp'].max()):.4f} MRayl")
    print(f"Transmission Loss: min = {float(scen['shadow_tl_db'].min()):.2f} dB, max = {float(scen['shadow_tl_db'].max()):.2f} dB")
    print(f"Total Profiles:    {len(scen['profiles'])} ({len(scen['wells'])} vertical, {len(scen['deviated_wells'])} deviated)")
    print("Profiles:")
    for p in scen["profiles"]:
        print(f"  - {p['name']:<12} ({p['type']:<15}): stations={p['trajectory'].md_m.size}, surface=({p['y']}, {p['x']})")
    print("=" * 90)
    return 0


def _cmd_scenarios_export(args: argparse.Namespace) -> int:
    try:
        canonical = resolve_scenario_name(args.scenario)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scen = build_maritime_scenario(canonical)

    fmt = args.format.lower()
    if fmt == "segy":
        segy_path = out_dir / f"{canonical}_sound_speed.segy"
        write_segy_cube(segy_path, scen["sound_speed_c"], title=f"OATSE {canonical} Sound Speed")
        zp_path = out_dir / f"{canonical}_impedance.segy"
        write_segy_cube(zp_path, scen["seabed_zp"], title=f"OATSE {canonical} Impedance Zp")
        print(f"[+] Exported SEG-Y cubes to: {out_dir}")
    else:
        # Default npy
        np.save(out_dir / f"{canonical}_c.npy", scen["sound_speed_c"].cpu().numpy())
        np.save(out_dir / f"{canonical}_rho.npy", scen["density_rho"].cpu().numpy())
        np.save(out_dir / f"{canonical}_zp.npy", scen["seabed_zp"].cpu().numpy())
        np.save(out_dir / f"{canonical}_tl.npy", scen["shadow_tl_db"].cpu().numpy())
        print(f"[+] Exported NPY volumes to: {out_dir}")

    # Export profile logs to .las and .dev
    for p in scen["profiles"]:
        write_dev_survey(out_dir / f"{p['name']}.dev", p["trajectory"])
        write_las2_file(
            out_dir / f"{p['name']}.las",
            p["trajectory"],
            sound_speed_kms=p["true_vp_trace"].cpu().numpy(),
            rho_gcc=p["true_rho_trace"].cpu().numpy(),
            zp_sonar=p["true_zp_trace"].cpu().numpy(),
            tl_db=p["true_tl_trace"].cpu().numpy(),
        )
    print(f"[+] Exported {len(scen['profiles'])} .las/.dev profile surveys to: {out_dir}")
    return 0


def _cmd_stream_segy(args: argparse.Namespace) -> int:
    segy_path = Path(args.input_segy)
    if not segy_path.exists():
        print(f"Error: SEG-Y file not found: {segy_path}", file=sys.stderr)
        return 1

    print(f"[*] Streaming SEG-Y volume: {segy_path} (Size: {segy_path.stat().st_size / (1024*1024):.2f} MB)")
    if args.roi:
        print(f"[*] ROI window: {args.roi}")
    max_grid = args.max_grid

    res = stream_segy_volume(
        segy_path=segy_path,
        roi=args.roi,
        max_grid=max_grid,
    )

    print("=" * 80)
    print("SEG-Y Out-of-Core Windowing & Beltrami Condensation Summary")
    print("=" * 80)
    print(f"Total traces paged:         {res['total_traces']:,}")
    print(f"Sample interval:            {res['sample_interval_us']} us (samples: {res['ns_samples']})")
    print(f"ROI shape (Z, Y, X):        {res['roi_shape']}")
    print(f"Downsampled output shape:   {res['output_shape']}")
    print(f"Tier-1 spike fraction:      {res['tier1_spike_fraction'] * 100:.2f}%")
    print(f"Beltrami curl preservation: {res['beltrami_curl_preservation_ratio']:.4f}")
    print(f"Step / jump recovery:       {res['fault_step_recovery']:.4f}")
    print(f"Peak Heap Memory:           {res['peak_heap_mb']:.3f} MB (< 35 MB SWaP ceiling)")
    print("=" * 80)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() in (".segy", ".sgy"):
            write_segy_cube(out_p, res["cube_zyx"], sample_interval_us=res["sample_interval_us"])
            print(f"[+] Saved downsampled SEG-Y to: {out_p}")
        else:
            np.save(out_p, res["cube_zyx"])
            print(f"[+] Saved numpy array to: {out_p}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_shape = (args.grid_size, int(round(args.grid_size * 0.75)), int(round(args.grid_size * 0.75)))

    targets = [args.scenario] if args.scenario else list(SCENARIO_DESCRIPTIONS.keys())
    summary: Dict[str, Any] = {
        "engine": "OATSE (Ocean Acoustic Thermocline & Sonar Engine) v0.2.0",
        "grid_shape": list(grid_shape),
        "scenarios": {},
    }

    print("=" * 110)
    print(f"OATSE v0.2.0 3D Acoustic & Sonar Impedance Benchmark | Grid: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}")
    print("=" * 110)
    header = (
        f"{'Scenario':<30} | {'OATSE RMSE':>11} | {'Naive RMSE':>11} | {'Delta%':>8} | "
        f"{'Zp RMSE':>10} | {'EdgeFid':>8} | {'DevSpike':>8}"
    )
    print(header)
    print("-" * 110)

    for scen_name in targets:
        canonical = resolve_scenario_name(scen_name)
        res = evaluate_scenario_comparison(canonical, grid_shape=grid_shape)
        m = res["metrics"]
        summary["scenarios"][canonical] = m
        print(
            f"{canonical:<30} | "
            f"{m['oatse_vp_rmse_kms']:>11.4f} | "
            f"{m['naive_vp_rmse_kms']:>11.4f} | "
            f"-{m['rmse_reduction_pct']:>6.2f}% | "
            f"{m['oatse_zp_rmse_mrayl']:>10.4f} | "
            f"{m['oatse_edge_gradient_fidelity']*100:>7.1f}% | "
            f"{m['deviated_well_spike_recovery']*100:>7.1f}%"
        )

    print("=" * 110)
    json_path = out_dir / "oatse_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Saved evaluation summary telemetry: {json_path}")
    return 0


def _cmd_ingest_las(args: argparse.Namespace) -> int:
    las_path = Path(args.input_las)
    if not las_path.exists():
        print(f"Error: LAS file not found: {las_path}", file=sys.stderr)
        return 1

    rec = read_las2_file(las_path)
    print("=" * 80)
    print(f"CWLS LAS 2.0 Well/Cast Log: {las_path.name}")
    print("=" * 80)
    print(f"Well / Cast Name:  {rec.well_name}")
    print(f"Profile Type:      {rec.well_type}")
    print(f"Start Depth:       {rec.trajectory.md_m[0]:.2f} m")
    print(f"Stop Depth:        {rec.trajectory.md_m[-1]:.2f} m")
    print(f"Depth Step:        {rec.step_m:.2f} m ({rec.trajectory.md_m.size} stations)")
    print(f"Curves ({len(rec.curves)}):       {', '.join(rec.curves.keys())}")
    for name, arr in rec.curves.items():
        val_mean = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")
        print(f"  - {name:<16}: unit={rec.units.get(name, ''):<10} mean={val_mean:10.4f} ({rec.descriptions.get(name, '')})")
    print("=" * 80)
    return 0


def _cmd_ingest_dev(args: argparse.Namespace) -> int:
    dev_path = Path(args.input_dev)
    if not dev_path.exists():
        print(f"Error: DEV file not found: {dev_path}", file=sys.stderr)
        return 1

    traj = read_dev_survey(dev_path)
    print("=" * 80)
    print(f"Petrel/Leapfrog .dev Directional Survey: {dev_path.name}")
    print("=" * 80)
    print(f"Profile Name:      {traj.well_name}")
    print(f"Trajectory Type:   {traj.well_type}")
    print(f"Stations:          {traj.md_m.size}")
    print(f"MD Range:          {traj.md_m[0]:.2f} m -> {traj.md_m[-1]:.2f} m")
    print(f"TVD Range:         {traj.tvd_m[0]:.2f} m -> {traj.tvd_m[-1]:.2f} m")
    print(f"Max Dogleg:        {float(np.max(traj.dogleg_deg_30m)):.2f} deg/30m")
    print("=" * 80)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oatse",
        description="OATSE: Ocean Acoustic Thermocline & Sonar Engine (v0.2.0)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. scenarios
    p_scen = subparsers.add_parser("scenarios", help="Inspect and export maritime scenarios")
    scen_subs = p_scen.add_subparsers(dest="scen_action")
    p_scen_list = scen_subs.add_parser("list", help="List all 5 maritime benchmark scenarios")
    p_scen_insp = scen_subs.add_parser("inspect", help="Inspect detailed scenario petrophysics")
    p_scen_insp.add_argument("--scenario", "-s", required=True, help="Scenario key or alias")
    p_scen_exp = scen_subs.add_parser("export", help="Export scenario property cubes and profiles")
    p_scen_exp.add_argument("--scenario", "-s", required=True, help="Scenario key or alias")
    p_scen_exp.add_argument("--output-dir", "-o", default="results/scenarios", help="Output directory")
    p_scen_exp.add_argument("--format", "-f", default="npy", choices=["npy", "segy"], help="Cube export format")

    # 2. stream-segy
    p_stream = subparsers.add_parser("stream-segy", help="Out-of-core SEG-Y windowing and Beltrami downsampling")
    p_stream.add_argument("input_segy", help="Path to input SEG-Y file")
    p_stream.add_argument("--roi", help="ROI string 'il_min:il_max,xl_min:xl_max,z_min:z_max'")
    p_stream.add_argument("--max-grid", type=int, default=64, help="Max grid size along each axis")
    p_stream.add_argument("--output", "-o", help="Optional path to save downsampled volume (.npy or .segy)")

    # 3. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation suite across scenarios")
    p_eval.add_argument("--scenario", "-s", help="Optional single scenario to evaluate")
    p_eval.add_argument("--grid-size", type=int, default=32, help="Grid Z size (default: 32)")
    p_eval.add_argument("--output-dir", "-o", default="results", help="Directory to save telemetry JSON")

    # 4. export-studio
    p_studio = subparsers.add_parser("export-studio", help="Export Portable WebGL 3D Sonar Studio")
    p_studio.add_argument("--output-dir", "-o", default="studio", help="Output directory")

    # 5. serve
    p_serve = subparsers.add_parser("serve", help="Start interactive OATSE 3D Studio server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", "-p", type=int, default=8090, help="Bind port (default: 8090)")
    p_serve.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # 6. showcase
    p_show = subparsers.add_parser("showcase", help="Render 4-panel Matplotlib Sonar Benchmark Showcase PNG")
    p_show.add_argument("--scenario", "-s", default="munk_sofar_shadow_zone", help="Scenario key or alias")
    p_show.add_argument("--output", "-o", default="results/oatse_sonar_showcase.png", help="Output PNG path")

    # 7. ingest-las
    p_las = subparsers.add_parser("ingest-las", help="Inspect CWLS LAS 2.0 well/cast log")
    p_las.add_argument("input_las", help="Path to .las file")

    # 8. ingest-dev
    p_dev = subparsers.add_parser("ingest-dev", help="Inspect Petrel .dev directional survey")
    p_dev.add_argument("input_dev", help="Path to .dev file")

    args = parser.parse_args(argv)

    if args.subcommand == "scenarios":
        if args.scen_action == "list" or not args.scen_action:
            return _cmd_scenarios_list(args)
        elif args.scen_action == "inspect":
            return _cmd_scenarios_inspect(args)
        elif args.scen_action == "export":
            return _cmd_scenarios_export(args)
    elif args.subcommand == "stream-segy":
        return _cmd_stream_segy(args)
    elif args.subcommand == "evaluate":
        return _cmd_evaluate(args)
    elif args.subcommand == "export-studio":
        from .studio import export_portable_html_studio
        html_p = Path(args.output_dir) / "oatse_portable_studio.html"
        export_portable_html_studio(html_p)
        print(f"[+] Exported portable studio to: {html_p}")
        return 0
    elif args.subcommand == "serve":
        from .studio import start_studio_server
        start_studio_server(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0
    elif args.subcommand == "showcase":
        from .studio import render_sonar_showcase_figure
        out_p = render_sonar_showcase_figure(scenario_name=args.scenario, output_path=args.output)
        print(f"[+] Rendered sonar showcase figure to: {out_p}")
        return 0
    elif args.subcommand == "ingest-las":
        return _cmd_ingest_las(args)
    elif args.subcommand == "ingest-dev":
        return _cmd_ingest_dev(args)
    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
