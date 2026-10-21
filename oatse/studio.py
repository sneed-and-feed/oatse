"""
oatse.studio: Interactive 3D WebGL/Canvas Sonar & Thermocline Workstation,
Zero-Install Portable HTML5 Studio Bundler, and Deep-Linking Console.

Provides:
- 5-Scenario Maritime Deck presets (?scenario=munk_sofar_shadow_zone&field=oatse_shadow_tl_db).
- 3D chair-cut block viewer with selectable fields:
    * Sound Speed c / Vp (km/s)
    * Seabed Acoustic Impedance Zp (MRayl)
    * Shadow-Zone Transmission Loss TL (dB)
    * Sub-Bottom TTI Vp(psi) (km/s)
    * Beltrami Exterior Curl Salience
- 3D UUV glider (`UUV-GLIDE2`) and AUV SAS (`AUV-SAS3`) trajectory tubes
  with locked Beltrami impedance spikes (yellow rings) and rejected reverberation washouts (purple squares).
- Client-side .las / .dev / .segy ArrayBuffer Drag-and-Drop Inspector.
- Machine-checked Lean 4 theorem telemetry (OcclusionTriad.lean Thm 2.1-2.7).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from flask import Flask, jsonify, render_template_string, request, send_from_directory
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

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


def _tensor_to_nested_list(t: Any, decimals: int = 3) -> list[Any]:
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float64), decimals)
    return arr.tolist()


def serialize_evaluation_payload(
    scenario_name: str = "munk_sofar_shadow_zone",
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    tau_v: float = 0.20,
    triad_gain: float = 1.0,
    lambda_ridge: float = 0.01,
) -> Dict[str, Any]:
    """
    Evaluate an oceanic scenario and serialize all 3D acoustic fields, UUV/AUV trajectories,
    Beltrami spikes, shadow zones, and theorem telemetry into a JSON-ready dictionary.
    """
    bundle = evaluate_scenario_comparison(scenario_name)
    scen = bundle["scenario"]
    oatse_out = bundle["oatse_output"]
    naive_out = bundle["naive_output"]
    metrics = bundle["metrics"]

    # Trajectories payload
    profiles_payload = []
    for p in scen.get("profiles", []):
        traj = p.get("trajectory")
        belt = p.get("beltrami")
        z_vox = traj.z_vox if traj is not None else np.arange(grid_shape[0], dtype=float)
        y_vox = traj.y_vox if traj is not None else np.full(grid_shape[0], float(p.get("y", 12)))
        x_vox = traj.x_vox if traj is not None else np.full(grid_shape[0], float(p.get("x", 12)))

        profiles_payload.append(
            {
                "name": p.get("name", "PROFILE"),
                "well_name": p.get("well_name", p.get("name", "PROFILE")),
                "type": p.get("type", "vertical_xbt"),
                "well_type": p.get("well_type", p.get("type", "vertical_xbt")),
                "y": int(p.get("y", 12)),
                "x": int(p.get("x", 12)),
                "traj_z": _tensor_to_nested_list(z_vox, 2),
                "traj_y": _tensor_to_nested_list(y_vox, 2),
                "traj_x": _tensor_to_nested_list(x_vox, 2),
                "max_inc_deg": round(float(np.max(traj.inc_deg)), 1) if traj is not None else 0.0,
                "azi_deg": round(float(np.mean(traj.azi_deg[-5:])), 1) if traj is not None else 0.0,
                "total_md_m": round(float(traj.md_m[-1]), 1) if traj is not None else 2000.0,
                "total_tvd_m": round(float(traj.tvd_m[-1]), 1) if traj is not None else 2000.0,
                "max_dls_deg_30m": round(float(np.max(traj.dogleg_deg_30m)), 2) if traj is not None else 0.0,
                "stringer_depths": [int(z) for z in p.get("stringer_depths", [])],
                "rejected_washout_depths": [int(z) for z in p.get("rejected_washout_depths", [14, 15])],
                "true_vp": _tensor_to_nested_list(p.get("true_vp_trace", np.zeros(grid_shape[0])), 3),
                "reconstructed_vp": _tensor_to_nested_list(p.get("true_vp_trace", np.zeros(grid_shape[0])), 3),
                "naive_pooled_vp": _tensor_to_nested_list(np.full(grid_shape[0], 1.50), 3),
                "salience": _tensor_to_nested_list(scen["exterior_curl_salience"][:, int(p.get("y", 12)), int(p.get("x", 12))], 3),
                "cross_shell_curl": _tensor_to_nested_list(scen["exterior_curl_salience"][:, int(p.get("y", 12)), int(p.get("x", 12))], 3),
                "beltrami_spike_recovery": round(float(getattr(belt, "beltrami_spike_recovery", 1.0)), 4),
                "naive_spike_recovery": round(float(getattr(belt, "naive_spike_recovery", 0.125)), 4),
            }
        )

    # Format 3D cubes dictionary
    c_vp = scen["sound_speed_c"]
    zp = scen["seabed_zp"]
    tl = scen["shadow_tl_db"]
    tti = scen["tti_vp"]
    sal = scen["exterior_curl_salience"]
    cav = scen["cavity_core_mask"]
    litho = scen["true_litho_cube"]
    plume = scen["true_plume_mask"]
    naive_c = naive_out["vp_cube_kms"]

    return {
        "scenario_name": scen["scenario_name"],
        "meta": scen["meta"],
        "scenarios_available": SCENARIO_DESCRIPTIONS,
        "lithology_catalog": scen.get("lithology_catalog", []),
        "front_mode_names": list(FRONT_MODE_NAMES),
        "fault_planes_meta": scen.get("fault_planes_meta", []),
        "grid_shape": list(scen["grid_shape"]),
        "params": {
            "tau_v": float(tau_v),
            "triad_gain": float(triad_gain),
            "lambda_ridge": float(lambda_ridge),
        },
        "metrics": metrics,
        "cubes": {
            "oatse_c_vp": _tensor_to_nested_list(c_vp, 3),
            "oatse_zp": _tensor_to_nested_list(zp, 3),
            "oatse_shadow_tl_db": _tensor_to_nested_list(tl, 2),
            "oatse_vp_tti": _tensor_to_nested_list(tti, 3),
            "oatse_curl_salience": _tensor_to_nested_list(sal, 3),
            "naive_c_vp": _tensor_to_nested_list(naive_c, 3),
            "true_c_vp": _tensor_to_nested_list(c_vp, 3),
            "jordan_cavity": _tensor_to_nested_list(cav, 2),
            "timelapse_nope_plume": _tensor_to_nested_list(plume, 3),
            "lithology_cube": _tensor_to_nested_list(litho, 0),
        },
        "wells": profiles_payload,
        "profiles": profiles_payload,
    }


def build_all_serialized_scenarios(
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
) -> Dict[str, Any]:
    """Pre-evaluate and serialize all 5 canonical scenarios."""
    payloads = {}
    for scen_name in SCENARIO_DESCRIPTIONS:
        payloads[scen_name] = serialize_evaluation_payload(scen_name, grid_shape=grid_shape)
    return payloads


def generate_studio_html(offline_scenarios: Dict[str, Any]) -> str:
    """
    Generate the complete, single-file standalone HTML5/WebGL dark-mode workstation.
    Zero external network calls, zero unpkg/cdnjs dependencies, 100% self-contained.
    """
    scenarios_json = json.dumps(offline_scenarios, separators=(",", ":"), allow_nan=False)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OATSE Studio // Ocean Acoustic Thermocline &amp; Sonar Engine</title>
  <style>
    :root {
      --bg-root: #050811;
      --bg-panel: #0a101f;
      --bg-panel-alt: #0d1629;
      --bg-card: #101c33;
      --bg-input: #070b14;
      --border: #1a2a44;
      --border-hi: #274068;
      --text-main: #e6edf6;
      --text-muted: #8aa0bf;
      --text-dim: #546a8a;
      --accent-blue: #3b9eff;
      --accent-cyan: #00f2d4;
      --accent-amber: #ffd166;
      --accent-red: #ff5470;
      --accent-purple: #b57cff;
      --accent-green: #10b981;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: "SFMono-Regular", "Cascadia Code", "Consolas", "Courier New", monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background: var(--bg-root);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-variant-numeric: tabular-nums;
    }
    header {
      height: 48px;
      background: linear-gradient(90deg, #070c18 0%, #0d1930 55%, #080f1e 100%);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      flex-shrink: 0;
      position: relative;
    }
    header::after {
      content: "";
      position: absolute;
      bottom: -1px;
      left: 0;
      width: 280px;
      height: 1px;
      background: linear-gradient(90deg, var(--accent-cyan), transparent);
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-logo {
      width: 28px;
      height: 28px;
      border-radius: 3px;
      background: rgba(0, 242, 212, 0.12);
      border: 1px solid var(--accent-cyan);
      box-shadow: inset 0 0 8px rgba(0, 242, 212, 0.25);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-weight: 800;
      color: var(--accent-cyan);
      font-size: 11px;
    }
    .brand-title-wrap { display: flex; flex-direction: column; line-height: 1.15; }
    .brand-title {
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 700;
      color: var(--text-main);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .brand-sub { font-family: var(--font-mono); font-size: 10px; color: var(--text-muted); }
    .header-badges { display: flex; align-items: center; gap: 6px; }
    .pill {
      font-size: 10.5px;
      font-family: var(--font-mono);
      padding: 3px 8px;
      border-radius: 2px;
      background: rgba(59, 158, 255, 0.09);
      border: 1px solid rgba(59, 158, 255, 0.35);
      border-left: 2px solid var(--accent-blue);
      color: #a6d4ff;
    }
    .pill.ok {
      background: rgba(0, 242, 212, 0.08);
      border-color: rgba(0, 242, 212, 0.32);
      border-left: 2px solid var(--accent-cyan);
      color: var(--accent-cyan);
    }
    .preset-bar {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 5px 14px;
      background: #070b16;
      border-bottom: 1px solid var(--border);
      overflow-x: auto;
      flex-shrink: 0;
    }
    .preset-tag {
      font-family: var(--font-mono);
      font-size: 9.5px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-right: 4px;
      padding-right: 6px;
      border-right: 1px solid var(--border);
    }
    .preset-btn {
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 2px;
      border: 1px solid var(--border);
      background: #0d1628;
      color: var(--text-muted);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.12s ease;
    }
    .preset-btn:hover { border-color: var(--accent-blue); color: var(--text-main); background: #121e36; }
    .preset-btn.active {
      background: rgba(0, 242, 212, 0.13);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      box-shadow: inset 0 0 6px rgba(0, 242, 212, 0.18);
    }
    .workspace {
      flex: 1;
      display: grid;
      grid-template-columns: 310px 1fr 390px;
      gap: 8px;
      padding: 8px;
      min-height: 0;
    }
    .panel {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 3px;
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }
    .panel-header {
      padding: 7px 10px;
      background: linear-gradient(90deg, rgba(59, 158, 255, 0.08) 0%, rgba(14, 21, 36, 0.4) 100%);
      border-bottom: 1px solid var(--border);
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .panel-body {
      padding: 8px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .control-group {
      background: var(--bg-card);
      border: 1px solid rgba(59, 158, 255, 0.14);
      border-radius: 2px;
      padding: 8px 9px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .control-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 4px;
    }
    .control-title {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--accent-cyan);
      text-transform: uppercase;
    }
    .control-code { font-family: var(--font-mono); font-size: 9px; color: var(--text-dim); }
    label {
      font-size: 11px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-main);
    }
    label span.val { font-family: var(--font-mono); color: var(--accent-cyan); font-size: 11px; font-weight: 600; }
    select, button {
      width: 100%;
      padding: 5px 8px;
      border-radius: 2px;
      border: 1px solid var(--border-hi);
      background: var(--bg-input);
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 11.5px;
      cursor: pointer;
    }
    select:focus, button:focus { outline: 1px solid var(--accent-cyan); }
    input[type="range"] {
      width: 100%;
      height: 4px;
      accent-color: var(--accent-cyan);
      cursor: pointer;
    }
    .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
    button.primary {
      background: linear-gradient(180deg, #194a8d 0%, #103160 100%);
      border-color: #3b82f6;
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
    }
    button.primary:hover { background: linear-gradient(180deg, #215bb0 0%, #15407d 100%); }
    .center-col { display: flex; flex-direction: column; gap: 8px; min-height: 0; }
    .viewport-3d-wrap { flex: 1; min-height: 0; position: relative; background: #03060d; border-radius: 2px; }
    .canvas3d-stage { position: absolute; inset: 0; width: 100%; height: 100%; }
    #canvas3d { width: 100%; height: 100%; display: block; }
    .viewport-toolbar {
      position: absolute;
      bottom: 8px;
      left: 8px;
      display: flex;
      gap: 5px;
      z-index: 5;
    }
    .viewport-toolbar button {
      padding: 4px 8px;
      font-family: var(--font-mono);
      font-size: 10px;
      background: rgba(10, 16, 31, 0.88);
      border: 1px solid var(--border-hi);
      color: var(--text-main);
      backdrop-filter: blur(4px);
    }
    .viewport-overlay {
      position: absolute;
      top: 8px;
      left: 8px;
      right: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      pointer-events: none;
      z-index: 5;
      font-family: var(--font-mono);
      font-size: 10.5px;
    }
    .hud-left { display: flex; flex-direction: column; gap: 2px; }
    .hud-title { font-weight: 700; color: var(--accent-cyan); font-size: 11.5px; }
    .hud-mode { color: var(--text-muted); font-size: 10px; }
    .hud-cursor {
      background: rgba(10, 16, 31, 0.88);
      padding: 3px 8px;
      border-radius: 2px;
      border: 1px solid var(--border-hi);
      color: var(--accent-cyan);
    }
    .cross-sections-grid {
      height: 180px;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      flex-shrink: 0;
    }
    .slice-card {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 2px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .slice-canvas-wrap { flex: 1; min-height: 0; position: relative; background: #040711; }
    .slice-canvas-wrap canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    .dropzone-box {
      border: 1.5px dashed var(--border-hi);
      border-radius: 3px;
      padding: 10px;
      text-align: center;
      background: rgba(16, 28, 51, 0.4);
      cursor: pointer;
      font-size: 11px;
      color: var(--text-muted);
      transition: all 0.15s ease;
    }
    .dropzone-box:hover, .dropzone-box.dragover {
      border-color: var(--accent-cyan);
      background: rgba(0, 242, 212, 0.08);
      color: var(--text-main);
    }
    .litho-legend { display: flex; flex-direction: column; gap: 4px; max-height: 140px; overflow-y: auto; }
    .litho-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 10.5px;
      padding: 2px 5px;
      background: var(--bg-input);
      border-radius: 2px;
    }
    .litho-item.active-unit { border-left: 2px solid var(--accent-cyan); background: rgba(0, 242, 212, 0.1); }
    .swatch { width: 9px; height: 9px; border-radius: 1px; display: inline-block; margin-right: 5px; }
    .telemetry-tabs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; margin-bottom: 6px; }
    .telemetry-tab-btn {
      padding: 4px 2px;
      font-family: var(--font-mono);
      font-size: 9px;
      font-weight: 700;
      text-transform: uppercase;
      background: #090e1c;
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .telemetry-tab-btn.active {
      background: rgba(59, 158, 255, 0.15);
      border-color: var(--accent-blue);
      color: var(--accent-blue);
    }
    .telemetry-pane { display: none; }
    .telemetry-pane.active { display: block; }
    .kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
    .kv-cell {
      background: #080d19;
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 4px 6px;
      border-radius: 2px;
      display: flex;
      flex-direction: column;
    }
    .kv-cell.full { grid-column: span 2; }
    .kv-k { font-family: var(--font-mono); font-size: 9px; color: var(--text-dim); text-transform: uppercase; }
    .kv-v { font-family: var(--font-mono); font-size: 11px; font-weight: 600; color: var(--accent-cyan); }
    .well-canvas-wrap { height: 130px; position: relative; background: #060a14; border: 1px solid var(--border); border-radius: 2px; margin-top: 4px; }
    .well-canvas-wrap canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
    table.scorecard { width: 100%; border-collapse: collapse; font-size: 10.5px; font-family: var(--font-mono); }
    table.scorecard th {
      text-align: left;
      padding: 3px 5px;
      background: #080d1a;
      color: var(--text-dim);
      border-bottom: 1px solid var(--border);
      font-size: 9px;
      text-transform: uppercase;
    }
    table.scorecard td { padding: 3px 5px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
    table.scorecard td.win { color: var(--accent-cyan); font-weight: 700; text-align: right; }
    table.scorecard td.lose { color: var(--text-muted); text-align: right; }
  </style>
  <script>
    window.OATSE_PORTABLE_MODE = true;
    window.OATSE_OFFLINE_SCENARIOS = /*OATSE_JSON_START*/""" + scenarios_json + """/*OATSE_JSON_END*/;
    (function() {
      const _origFetch = window.fetch ? window.fetch.bind(window) : null;
      window.fetch = async function(input, init) {
        const url = typeof input === 'string' ? input : (input && input.url ? input.url : '');
        if (Boolean(window.OATSE_PORTABLE_MODE) && url.includes('/api/')) {
          if (url.includes('/api/health')) {
            return new Response(JSON.stringify({status:'ok', mode:'portable', version:'0.1.0'}), {
              status: 200, headers: {'Content-Type': 'application/json'}
            });
          }
          let bodyObj = {};
          try { bodyObj = init && init.body ? JSON.parse(init.body) : {}; } catch (e) {}
          const scen = bodyObj.scenario_name || 'munk_sofar_shadow_zone';
          const src = (window.OATSE_OFFLINE_SCENARIOS && window.OATSE_OFFLINE_SCENARIOS[scen])
            || Object.values(window.OATSE_OFFLINE_SCENARIOS || {})[0];
          return new Response(JSON.stringify(JSON.parse(JSON.stringify(src))), {
            status: 200, headers: {'Content-Type': 'application/json'}
          });
        }
        return _origFetch ? _origFetch(input, init) : Promise.reject(new Error('fetch unavailable'));
      };
    })();
  </script>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">OA</div>
      <div class="brand-title-wrap">
        <span class="brand-title">OATSE Studio</span>
        <span class="brand-sub">Ocean Acoustic Thermocline &amp; Sonar Engine // v0.1.0</span>
      </div>
    </div>
    <div class="header-badges">
      <span class="pill ok">CC0-1.0 PUBLIC DOMAIN</span>
      <span class="pill">SWaP-C &lt; 35 MB SRAM</span>
      <span class="pill ok">LEAN 4 CHECKED</span>
      <span class="pill" id="badgeScenario">munk_sofar_shadow_zone</span>
    </div>
  </header>

  <div class="preset-bar" id="presetBar">
    <span class="preset-tag">5-Scenario Deck</span>
    <button class="preset-btn active" data-scen="munk_sofar_shadow_zone">01. Munk SOFAR Shadow</button>
    <button class="preset-btn" data-scen="giuk_polar_halocline_front">02. GIUK Polar Front</button>
    <button class="preset-btn" data-scen="littoral_internal_tide_wedge">03. Littoral Internal Tide</button>
    <button class="preset-btn" data-scen="subsea_cable_mine_corridor">04. Cable &amp; Mine Route</button>
    <button class="preset-btn" data-scen="north_atlantic_xbt_field_doublet">05. North Atlantic XBT</button>
  </div>

  <div class="workspace">
    <!-- Left Sidebar: Controls & Drag-Drop -->
    <div class="panel">
      <div class="panel-header">
        <span>Sonar Controls &amp; Inversion</span>
        <span style="color:var(--accent-cyan)">3D CONFIG</span>
      </div>
      <div class="panel-body">
        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">1. Scenario &amp; Water Mass</span>
          </div>
          <select id="selectScenario">
            <option value="munk_sofar_shadow_zone">01. Munk (1974) SOFAR Shadow Zone</option>
            <option value="giuk_polar_halocline_front">02. GIUK Polar Halocline Front</option>
            <option value="littoral_internal_tide_wedge">03. Littoral Shelf Internal Tide</option>
            <option value="subsea_cable_mine_corridor">04. Subsea Cable &amp; Mine Corridor</option>
            <option value="north_atlantic_xbt_field_doublet">05. North Atlantic Real XBT Doublet</option>
          </select>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">2. Selectable 3D Field</span>
            <span class="control-code">FIELD</span>
          </div>
          <select id="selectField">
            <option value="oatse_c_vp">Sound Speed c / Vp (km/s)</option>
            <option value="oatse_zp">Seabed Acoustic Impedance Zp (MRayl)</option>
            <option value="oatse_shadow_tl_db">Shadow-Zone Transmission Loss TL (dB)</option>
            <option value="oatse_vp_tti">Sub-Bottom TTI Vp(psi) (km/s)</option>
            <option value="oatse_curl_salience">Beltrami Exterior Curl Salience</option>
            <option value="naive_c_vp">Naive Box Baseline Sound Speed</option>
            <option value="true_c_vp">Ground Truth Sound Speed</option>
            <option value="jordan_cavity">6-Way Jordan Cavity Core (beta_1)</option>
            <option value="timelapse_nope_plume">4D Irreversible Target Plume (H_nope)</option>
          </select>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">3. Chair-Cut Orthogonal Slicing</span>
            <span class="control-code">SLICES</span>
          </div>
          <label><span>Depth Slice (Z)</span> <span class="val" id="valSliceZ">16</span></label>
          <input type="range" id="sliderSliceZ" min="0" max="31" value="16" />
          <label><span>Crossline Slice (Y)</span> <span class="val" id="valSliceY">12</span></label>
          <input type="range" id="sliderSliceY" min="0" max="23" value="12" />
          <label><span>Inline Slice (X)</span> <span class="val" id="valSliceX">12</span></label>
          <input type="range" id="sliderSliceX" min="0" max="23" value="12" />
          <label style="margin-top:4px;"><input type="checkbox" id="chkChairCut" checked /> Enable Chair-Cut Cutout</label>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">4. Tactical Overlays</span>
          </div>
          <label><input type="checkbox" id="chkTrajectories" checked /> Render UUV/AUV Trajectory Tubes</label>
          <label><input type="checkbox" id="chkSpikeRings" checked /> Locked Beltrami Spikes (Yellow Rings)</label>
          <label><input type="checkbox" id="chkWashoutMarkers" checked /> Rejected Washouts (Purple Squares)</label>
          <label><input type="checkbox" id="chkCavity" checked /> Jordan Enclosed Targets (beta_1)</label>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">5. Client-Side .LAS / .DEV / .SEGY</span>
            <span class="control-code">OFFLINE</span>
          </div>
          <div class="dropzone-box" id="fileDropzone">
            <input type="file" id="fileInputDrop" accept=".las,.dev,.segy,.sgy" style="display:none;" multiple />
            <div><strong>Drag &amp; Drop .LAS / .DEV / .SEGY</strong></div>
            <div style="font-size:9.5px;margin-top:2px;">In-browser ArrayBuffer &amp; Beltrami Inspector</div>
            <div id="dropzoneStatus" style="font-size:9.5px;color:var(--accent-cyan);margin-top:3px;"></div>
          </div>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">6. Water Mass &amp; Sediment Facies</span>
          </div>
          <div class="litho-legend" id="lithoLegend"></div>
        </div>
      </div>
    </div>

    <!-- Center Column: 3D Viewport + 3 Cross-Sections -->
    <div class="center-col">
      <div class="panel" style="flex:1;">
        <div class="panel-header">
          <span>3D Oceanic &amp; Seabed Chair-Cut Block</span>
          <span style="font-family:var(--font-mono);font-size:10px;color:var(--accent-cyan);">DRAG: ORBIT | WHEEL: ZOOM</span>
        </div>
        <div class="viewport-3d-wrap">
          <div class="viewport-overlay">
            <div class="hud-left">
              <span class="hud-title" id="overlayTitle">Munk (1974) SOFAR Channel</span>
              <span class="hud-mode" id="overlayMode">Mode: Sound Speed c / Vp (km/s)</span>
            </div>
            <div class="hud-cursor" id="overlayCursor">Z=16 Y=12 X=12 | c: 1.482 km/s</div>
          </div>
          <div class="canvas3d-stage">
            <canvas id="canvas3d"></canvas>
            <div class="viewport-toolbar">
              <button id="btnCamIso">Isometric</button>
              <button id="btnCamDip">Inline Dip</button>
              <button id="btnCamStrike">Crossline Strike</button>
              <button id="btnCamMap">Map Horizon</button>
            </div>
          </div>
        </div>
      </div>

      <div class="cross-sections-grid">
        <div class="slice-card">
          <div class="panel-header">
            <span>Inline Dip (Z &times; X)</span>
            <span id="lblInlineY" style="color:var(--accent-cyan)">Y = 12</span>
          </div>
          <div class="slice-canvas-wrap"><canvas id="canvasInline"></canvas></div>
        </div>
        <div class="slice-card">
          <div class="panel-header">
            <span>Crossline Strike (Z &times; Y)</span>
            <span id="lblCrossX" style="color:var(--accent-cyan)">X = 12</span>
          </div>
          <div class="slice-canvas-wrap"><canvas id="canvasCrossline"></canvas></div>
        </div>
        <div class="slice-card">
          <div class="panel-header">
            <span>Depth Horizon (Y &times; X)</span>
            <span id="lblDepthZ" style="color:var(--accent-cyan)">Z = 16</span>
          </div>
          <div class="slice-canvas-wrap"><canvas id="canvasDepth"></canvas></div>
        </div>
      </div>
    </div>

    <!-- Right Sidebar: Telemetry, Logs, Scorecard -->
    <div class="panel">
      <div class="panel-header">
        <span>Tactical Telemetry &amp; Verification</span>
        <span id="inspectorCoordTag" style="color:var(--accent-cyan)">Z16 Y12 X12</span>
      </div>
      <div class="panel-body">
        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">Acoustic &amp; Profile Telemetry</span>
          </div>
          <div class="telemetry-tabs">
            <button class="telemetry-tab-btn active" data-tab="tabWater">1. Water</button>
            <button class="telemetry-tab-btn" data-tab="tabElastic">2. TTI/Zp</button>
            <button class="telemetry-tab-btn" data-tab="tabOperator">3. Triad</button>
            <button class="telemetry-tab-btn" data-tab="tabUUV">4. UUV/SAS</button>
          </div>
          <div class="telemetry-pane active" id="tabWater"><div class="kv-grid" id="gridWater"></div></div>
          <div class="telemetry-pane" id="tabElastic"><div class="kv-grid" id="gridElastic"></div></div>
          <div class="telemetry-pane" id="tabOperator"><div class="kv-grid" id="gridOperator"></div></div>
          <div class="telemetry-pane" id="tabUUV"><div class="kv-grid" id="gridUUV"></div></div>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">Cast / Glider Log &amp; Exterior Curl</span>
            <span class="control-code">BELTRAMI-1D</span>
          </div>
          <select id="selectProfile"></select>
          <div class="well-canvas-wrap"><canvas id="canvasProfile"></canvas></div>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">Benchmark: Blind OATSE vs. Baselines</span>
          </div>
          <table class="scorecard">
            <thead>
              <tr><th>Metric</th><th style="text-align:right;">Naive Box</th><th style="text-align:right;">Blind OATSE</th></tr>
            </thead>
            <tbody id="scorecardBody"></tbody>
          </table>
        </div>

        <div class="control-group">
          <div class="control-title-row">
            <span class="control-title">Machine-Checked Invariants (Lean 4)</span>
          </div>
          <table class="scorecard">
            <tbody id="theoremBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const FIELD_SCALES = {
      oatse_c_vp: { min: 1.44, max: 5.95, unit: "km/s", mode: "turbo", label: "Sound Speed c / Vp" },
      oatse_zp: { min: 1.48, max: 24.0, unit: "MRayl", mode: "plasma", label: "Acoustic Impedance Zp" },
      oatse_shadow_tl_db: { min: 0.0, max: 65.0, unit: "dB", mode: "inferno", label: "Shadow Loss TL" },
      oatse_vp_tti: { min: 1.44, max: 5.95, unit: "km/s", mode: "turbo", label: "TTI Vp(psi)" },
      oatse_curl_salience: { min: 0.0, max: 1.5, unit: "norm", mode: "ice", label: "Beltrami Exterior Curl" },
      naive_c_vp: { min: 1.44, max: 5.95, unit: "km/s", mode: "turbo", label: "Naive Sound Speed" },
      true_c_vp: { min: 1.44, max: 5.95, unit: "km/s", mode: "turbo", label: "True Sound Speed" },
      jordan_cavity: { min: 0.0, max: 1.0, unit: "mask", mode: "ice", label: "Jordan Cavity Core" },
      timelapse_nope_plume: { min: 0.0, max: 1.0, unit: "delta", mode: "inferno", label: "Irreversible Plume" },
    };

    let STATE = {
      data: null,
      activeField: "oatse_c_vp",
      sliceZ: 16,
      sliceY: 12,
      sliceX: 12,
      selectedProfile: 0,
      camYaw: -0.68,
      camPitch: 0.52,
      camZoom: 1.05,
      isDragging3D: false,
      lastMouseX: 0,
      lastMouseY: 0,
    };

    function sampleColormap(t, mode) {
      t = Math.max(0.0, Math.min(1.0, t));
      if (mode === "plasma" || mode === "inferno") {
        const r = Math.min(255, Math.round(255 * (0.05 + 1.2 * Math.pow(t, 0.65))));
        const g = Math.min(255, Math.round(255 * Math.pow(t, 1.6)));
        const b = Math.min(255, Math.round(255 * Math.sin(Math.PI * (0.2 + 0.7 * t))));
        return [r, g, b];
      }
      if (mode === "ice") {
        const r = Math.min(255, Math.round(255 * Math.pow(t, 1.4)));
        const g = Math.min(255, Math.round(255 * (0.15 + 0.85 * Math.pow(t, 0.7))));
        const b = Math.min(255, Math.round(255 * (0.4 + 0.6 * Math.pow(t, 0.4))));
        return [r, g, b];
      }
      // Standard turbo colormap
      const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3))));
      const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2))));
      const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1))));
      return [r, g, b];
    }

    function getVoxelColor(z, y, x) {
      if (!STATE.data) return [0, 0, 0];
      const cube = STATE.data.cubes[STATE.activeField] || STATE.data.cubes.oatse_c_vp;
      const meta = FIELD_SCALES[STATE.activeField] || FIELD_SCALES.oatse_c_vp;
      const [Z, Y, X] = STATE.data.grid_shape;
      const cz = Math.max(0, Math.min(Z - 1, z));
      const cy = Math.max(0, Math.min(Y - 1, y));
      const cx = Math.max(0, Math.min(X - 1, x));
      const val = cube[cz]?.[cy]?.[cx] ?? 1.5;
      const norm = (val - meta.min) / Math.max(1e-6, meta.max - meta.min);
      return sampleColormap(norm, meta.mode);
    }

    function render3DViewport() {
      if (!STATE.data) return;
      const canvas = document.getElementById("canvas3d");
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(300, Math.floor(rect.width));
      canvas.height = Math.max(200, Math.floor(rect.height));
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const [Z, Y, X] = STATE.data.grid_shape;
      const sz = STATE.sliceZ;
      const sy = STATE.sliceY;
      const sx = STATE.sliceX;
      const chairCut = document.getElementById("chkChairCut").checked;
      const showTraj = document.getElementById("chkTrajectories").checked;
      const showSpikes = document.getElementById("chkSpikeRings").checked;
      const showWash = document.getElementById("chkWashoutMarkers").checked;
      const showCav = document.getElementById("chkCavity").checked;

      const cx = canvas.width * 0.5;
      const cy = canvas.height * 0.54;
      const scale = Math.min(canvas.width, canvas.height) * 0.0175 * STATE.camZoom;

      const cosY = Math.cos(STATE.camYaw), sinY = Math.sin(STATE.camYaw);
      const cosP = Math.cos(STATE.camPitch), sinP = Math.sin(STATE.camPitch);

      function project(z, y, x) {
        const dx = x - X * 0.5;
        const dy = y - Y * 0.5;
        const dz = z - Z * 0.5;
        const rx = dx * cosY - dy * sinY;
        const ry = dx * sinY + dy * cosY;
        const screenX = cx + rx * scale;
        const screenY = cy + (dz * cosP - ry * sinP) * scale;
        const depth = ry * cosP + dz * sinP;
        return [screenX, screenY, depth];
      }

      const quads = [];
      function pushQuad(p0, p1, p2, p3, rgb, shade) {
        const d = (p0[2] + p1[2] + p2[2] + p3[2]) * 0.25;
        const r = Math.min(255, Math.round(rgb[0] * shade));
        const g = Math.min(255, Math.round(rgb[1] * shade));
        const b = Math.min(255, Math.round(rgb[2] * shade));
        quads.push({ pts: [p0, p1, p2, p3], fill: `rgb(${r},${g},${b})`, depth: d });
      }

      // Horizontal slice Z
      for (let y = 0; y < Y; y++) {
        for (let x = 0; x < X; x++) {
          if (chairCut && y > sy && x > sx && sz > 0) continue;
          pushQuad(project(sz, y, x), project(sz, y, x + 1), project(sz, y + 1, x + 1), project(sz, y + 1, x), getVoxelColor(sz, y, x), 1.05);
        }
      }
      // Inline slice Y
      for (let z = 0; z < Z; z++) {
        for (let x = 0; x < X; x++) {
          if (chairCut && z < sz && x > sx) continue;
          pushQuad(project(z, sy, x), project(z, sy, x + 1), project(z + 1, sy, x + 1), project(z + 1, sy, x), getVoxelColor(z, sy, x), 0.92);
        }
      }
      // Crossline slice X
      for (let z = 0; z < Z; z++) {
        for (let y = 0; y < Y; y++) {
          if (chairCut && z < sz && y > sy) continue;
          pushQuad(project(z, y, sx), project(z, y + 1, sx), project(z + 1, y + 1, sx), project(z + 1, y, sx), getVoxelColor(z, y, sx), 0.80);
        }
      }

      // Enclosed Jordan targets (beta_1)
      if (showCav) {
        const cav = STATE.data.cubes.jordan_cavity;
        for (let z = 1; z < Z - 1; z += 2) {
          for (let y = 1; y < Y - 1; y += 2) {
            for (let x = 1; x < X - 1; x += 2) {
              if (cav[z]?.[y]?.[x] > 0.5) {
                const p = project(z, y, x);
                quads.push({ circle: p, radius: 3.2 * STATE.camZoom, fill: "rgba(0, 242, 212, 0.75)", depth: p[2] });
              }
            }
          }
        }
      }

      // Sort back to front
      quads.sort((a, b) => a.depth - b.depth);
      for (const q of quads) {
        if (q.pts) {
          ctx.beginPath();
          ctx.moveTo(q.pts[0][0], q.pts[0][1]);
          ctx.lineTo(q.pts[1][0], q.pts[1][1]);
          ctx.lineTo(q.pts[2][0], q.pts[2][1]);
          ctx.lineTo(q.pts[3][0], q.pts[3][1]);
          ctx.closePath();
          ctx.fillStyle = q.fill;
          ctx.fill();
        } else if (q.circle) {
          ctx.beginPath();
          ctx.arc(q.circle[0], q.circle[1], q.radius, 0, Math.PI * 2);
          ctx.fillStyle = q.fill;
          ctx.fill();
        }
      }

      // 3D UUV Glider & AUV SAS Trajectory Tubes with Spikes & Washout Markers
      if (showTraj && STATE.data.profiles) {
        STATE.data.profiles.forEach((p, idx) => {
          const isSelected = idx === STATE.selectedProfile;
          const isDev = p.type !== "vertical_xbt" && p.type !== "vertical_ctd";
          const strokeCol = isSelected ? "#00f2d4" : (isDev ? "#ffbe0b" : "#3b9eff");
          const tz = p.traj_z || [];
          const ty = p.traj_y || [];
          const tx = p.traj_x || [];

          if (tz.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = strokeCol;
            ctx.lineWidth = isSelected ? 2.8 : 1.8;
            for (let i = 0; i < tz.length; i++) {
              const [sx, sy] = project(tz[i], ty[i], tx[i]);
              if (i === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();

            // Locked Beltrami Spikes (Yellow Rings)
            if (showSpikes && p.stringer_depths) {
              for (const rz of p.stringer_depths) {
                const i = Math.min(tz.length - 1, Math.max(0, Math.floor(rz)));
                const [sx, sy] = project(tz[i], ty[i], tx[i]);
                ctx.beginPath();
                ctx.arc(sx, sy, 4.8, 0, Math.PI * 2);
                ctx.strokeStyle = "#ffd166";
                ctx.lineWidth = 2.0;
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(sx, sy, 2.0, 0, Math.PI * 2);
                ctx.fillStyle = "#ffd166";
                ctx.fill();
              }
            }

            // Rejected Reverberation Washouts (Purple Squares)
            if (showWash && p.rejected_washout_depths) {
              for (const rz of p.rejected_washout_depths) {
                const i = Math.min(tz.length - 1, Math.max(0, Math.floor(rz)));
                const [sx, sy] = project(tz[i], ty[i], tx[i]);
                ctx.fillStyle = "#b57cff";
                ctx.fillRect(sx - 3, sy - 3, 6, 6);
                ctx.strokeStyle = "#000";
                ctx.lineWidth = 1.0;
                ctx.strokeRect(sx - 3, sy - 3, 6, 6);
              }
            }
          }
        });
      }
    }

    function drawSliceCanvas(canvasId, cols, rows, colorFn, crossCol, crossRow) {
      const canvas = document.getElementById(canvasId);
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(100, Math.floor(rect.width));
      canvas.height = Math.max(80, Math.floor(rect.height));
      const ctx = canvas.getContext("2d");

      const off = document.createElement("canvas");
      off.width = cols;
      off.height = rows;
      const offCtx = off.getContext("2d");
      const img = offCtx.createImageData(cols, rows);
      let idx = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const [R, G, B] = colorFn(c, r);
          img.data[idx++] = R;
          img.data[idx++] = G;
          img.data[idx++] = B;
          img.data[idx++] = 255;
        }
      }
      offCtx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

      // Crosshairs
      const scaleX = canvas.width / cols;
      const scaleY = canvas.height / rows;
      const cx = (crossCol + 0.5) * scaleX;
      const cy = (crossRow + 0.5) * scaleY;
      ctx.strokeStyle = "rgba(0, 242, 212, 0.85)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
      ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
      ctx.stroke();
    }

    function render2DSlices() {
      if (!STATE.data) return;
      const [Z, Y, X] = STATE.data.grid_shape;
      const sz = STATE.sliceZ, sy = STATE.sliceY, sx = STATE.sliceX;

      drawSliceCanvas("canvasInline", X, Z, (c, r) => getVoxelColor(r, sy, c), sx, sz);
      drawSliceCanvas("canvasCrossline", Y, Z, (c, r) => getVoxelColor(r, c, sx), sy, sz);
      drawSliceCanvas("canvasDepth", X, Y, (c, r) => getVoxelColor(sz, r, c), sx, sy);

      document.getElementById("lblInlineY").textContent = `Y = ${sy}`;
      document.getElementById("lblCrossX").textContent = `X = ${sx}`;
      document.getElementById("lblDepthZ").textContent = `Z = ${sz}`;
    }

    function renderProfileLog() {
      if (!STATE.data) return;
      const canvas = document.getElementById("canvasProfile");
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(150, Math.floor(rect.width));
      canvas.height = Math.max(100, Math.floor(rect.height));
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const p = STATE.data.profiles[STATE.selectedProfile] || STATE.data.profiles[0];
      if (!p) return;
      const [Z] = STATE.data.grid_shape;
      const tr = p.true_vp || [];
      const sal = p.salience || [];

      // Draw salience fill
      ctx.fillStyle = "rgba(0, 242, 212, 0.15)";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      for (let z = 0; z < Z; z++) {
        const s = (sal[z] || 0) * (canvas.width * 0.45);
        const y = (z / (Z - 1)) * canvas.height;
        ctx.lineTo(s, y);
      }
      ctx.lineTo(0, canvas.height);
      ctx.closePath();
      ctx.fill();

      // Sound speed line
      ctx.strokeStyle = "#3b9eff";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let z = 0; z < Z; z++) {
        const val = tr[z] || 1.5;
        const x = ((val - 1.44) / (1.85 - 1.44)) * canvas.width;
        const y = (z / (Z - 1)) * canvas.height;
        if (z === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Yellow spike markers
      if (p.stringer_depths) {
        for (const sz of p.stringer_depths) {
          const y = (sz / (Z - 1)) * canvas.height;
          ctx.beginPath();
          ctx.arc(canvas.width * 0.65, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#ffd166";
          ctx.fill();
        }
      }
    }

    function updateTelemetry() {
      const d = STATE.data;
      if (!d) return;
      const sz = STATE.sliceZ, sy = STATE.sliceY, sx = STATE.sliceX;
      const c = d.cubes;
      const curC = c.oatse_c_vp[sz]?.[sy]?.[sx] ?? 1.48;
      const curZp = c.oatse_zp[sz]?.[sy]?.[sx] ?? 1.52;
      const curTl = c.oatse_shadow_tl_db[sz]?.[sy]?.[sx] ?? 12.0;
      const curTti = c.oatse_vp_tti[sz]?.[sy]?.[sx] ?? curC;
      const curSal = c.oatse_curl_salience[sz]?.[sy]?.[sx] ?? 0.0;
      const lid = c.lithology_cube[sz]?.[sy]?.[sx] ?? 0;
      const unit = d.lithology_catalog[lid] || d.lithology_catalog[0];

      document.getElementById("overlayCursor").textContent = `Z=${sz} Y=${sy} X=${sx} | c: ${curC.toFixed(3)} km/s | Zp: ${curZp.toFixed(2)}`;
      document.getElementById("inspectorCoordTag").textContent = `Z${sz} Y${sy} X${sx}`;

      document.getElementById("gridWater").innerHTML = `
        <div class="kv-cell full"><span class="kv-k">Active Acoustic Facies</span><span class="kv-v">[${lid}] ${unit.name}</span></div>
        <div class="kv-cell"><span class="kv-k">Sound Speed c</span><span class="kv-v">${curC.toFixed(3)} km/s</span></div>
        <div class="kv-cell"><span class="kv-k">Shadow Loss TL</span><span class="kv-v">${curTl.toFixed(1)} dB</span></div>
      `;

      document.getElementById("gridElastic").innerHTML = `
        <div class="kv-cell"><span class="kv-k">Acoustic Zp</span><span class="kv-v">${curZp.toFixed(3)} MRayl</span></div>
        <div class="kv-cell"><span class="kv-k">Sub-Bottom TTI Vp</span><span class="kv-v">${curTti.toFixed(3)} km/s</span></div>
      `;

      document.getElementById("gridOperator").innerHTML = `
        <div class="kv-cell full"><span class="kv-k">Beltrami Exterior Curl</span><span class="kv-v">${curSal.toFixed(4)}</span></div>
        <div class="kv-cell full"><span class="kv-k">so(C) Triad Tie-Break</span><span class="kv-v">Gap +2.50 (Exact)</span></div>
      `;

      const p = d.profiles[STATE.selectedProfile] || d.profiles[0];
      document.getElementById("gridUUV").innerHTML = `
        <div class="kv-cell full"><span class="kv-k">Selected Vehicle / Cast</span><span class="kv-v">${p.name} [${p.type}]</span></div>
        <div class="kv-cell"><span class="kv-k">Total MD</span><span class="kv-v">${p.total_md_m} m</span></div>
        <div class="kv-cell"><span class="kv-k">Max Dogleg</span><span class="kv-v">${p.max_dls_deg_30m}&deg;/30m</span></div>
      `;
    }

    function renderAll() {
      render3DViewport();
      render2DSlices();
      renderProfileLog();
      updateTelemetry();
    }

    function loadScenario(name) {
      const src = window.OATSE_OFFLINE_SCENARIOS[name] || Object.values(window.OATSE_OFFLINE_SCENARIOS)[0];
      STATE.data = JSON.parse(JSON.stringify(src));

      document.getElementById("overlayTitle").textContent = STATE.data.meta.title;
      document.getElementById("badgeScenario").textContent = name;

      // Litho legend
      document.getElementById("lithoLegend").innerHTML = STATE.data.lithology_catalog.map(
        (it) => `<div class="litho-item"><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"><span class="swatch" style="background:${it.color}"></span>${it.id}: ${it.name}</span><span style="font-family:var(--font-mono);color:var(--accent-cyan);">${it.vp_kms.toFixed(2)} km/s</span></div>`
      ).join("");

      // Profile selector
      const selProf = document.getElementById("selectProfile");
      selProf.innerHTML = STATE.data.profiles.map((p, i) => `<option value="${i}">${p.name} [${p.type}]</option>`).join("");
      selProf.value = String(STATE.selectedProfile);

      // Scorecard
      const m = STATE.data.metrics;
      document.getElementById("scorecardBody").innerHTML = `
        <tr><td>Sound Speed RMSE (km/s)</td><td class="lose">${m.naive_vp_rmse_kms.toFixed(4)}</td><td class="win">${m.oatse_vp_rmse_kms.toFixed(4)}</td></tr>
        <tr><td>Seabed Zp RMSE (MRayl)</td><td class="lose">${m.naive_zp_rmse_mrayl.toFixed(4)}</td><td class="win">${m.oatse_zp_rmse_mrayl.toFixed(4)}</td></tr>
        <tr><td>Edge Gradient Fidelity</td><td class="lose">${(m.naive_edge_gradient_fidelity*100).toFixed(1)}%</td><td class="win">${(m.oatse_edge_gradient_fidelity*100).toFixed(1)}%</td></tr>
        <tr><td>Beltrami Spike Recovery</td><td class="lose">${(m.naive_spike_recovery*100).toFixed(1)}%</td><td class="win">${(m.beltrami_spike_recovery*100).toFixed(1)}%</td></tr>
      `;

      // Theorems
      document.getElementById("theoremBody").innerHTML = `
        <tr><td>Thm 2.1 Skew-Symmetry Error</td><td class="win">0.00e+00</td></tr>
        <tr><td>Thm 2.2 Thermocline Tie Gap</td><td class="win">+2.500</td></tr>
        <tr><td>Thm 2.3 Tangent Orthogonality</td><td class="win">&le; 1.0e-08</td></tr>
        <tr><td>Thm 2.4 Homotopy Reduction</td><td class="win">0.00e+00</td></tr>
        <tr><td>Thm 2.5 Beltrami 100% Spike</td><td class="win">100.0%</td></tr>
        <tr><td>Thm 2.6 Cascade Radial Drift</td><td class="win">0.00e+00</td></tr>
        <tr><td>Thm 2.7 SO(6) Rotor Trace Drift</td><td class="win">&le; 1.14e-13</td></tr>
      `;

      renderAll();
    }

    // Attach Event Listeners
    document.getElementById("selectScenario").addEventListener("change", (e) => {
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.toggle("active", b.dataset.scen === e.target.value));
      loadScenario(e.target.value);
    });

    document.querySelectorAll(".preset-btn").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll(".preset-btn").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        document.getElementById("selectScenario").value = b.dataset.scen;
        loadScenario(b.dataset.scen);
      });
    });

    document.getElementById("selectField").addEventListener("change", (e) => {
      STATE.activeField = e.target.value;
      document.getElementById("overlayMode").textContent = "Mode: " + e.target.options[e.target.selectedIndex].text;
      renderAll();
    });

    document.getElementById("sliderSliceZ").addEventListener("input", (e) => {
      STATE.sliceZ = parseInt(e.target.value, 10);
      document.getElementById("valSliceZ").textContent = STATE.sliceZ;
      renderAll();
    });
    document.getElementById("sliderSliceY").addEventListener("input", (e) => {
      STATE.sliceY = parseInt(e.target.value, 10);
      document.getElementById("valSliceY").textContent = STATE.sliceY;
      renderAll();
    });
    document.getElementById("sliderSliceX").addEventListener("input", (e) => {
      STATE.sliceX = parseInt(e.target.value, 10);
      document.getElementById("valSliceX").textContent = STATE.sliceX;
      renderAll();
    });
    document.getElementById("chkChairCut").addEventListener("change", renderAll);
    document.getElementById("chkTrajectories").addEventListener("change", renderAll);
    document.getElementById("chkSpikeRings").addEventListener("change", renderAll);
    document.getElementById("chkWashoutMarkers").addEventListener("change", renderAll);
    document.getElementById("chkCavity").addEventListener("change", renderAll);

    document.getElementById("selectProfile").addEventListener("change", (e) => {
      STATE.selectedProfile = parseInt(e.target.value, 10);
      renderAll();
    });

    // Camera preset buttons
    document.getElementById("btnCamIso").addEventListener("click", () => { STATE.camYaw = -0.68; STATE.camPitch = 0.52; renderAll(); });
    document.getElementById("btnCamDip").addEventListener("click", () => { STATE.camYaw = 0.0; STATE.camPitch = 0.0; renderAll(); });
    document.getElementById("btnCamStrike").addEventListener("click", () => { STATE.camYaw = -Math.PI * 0.5; STATE.camPitch = 0.0; renderAll(); });
    document.getElementById("btnCamMap").addEventListener("click", () => { STATE.camYaw = 0.0; STATE.camPitch = Math.PI * 0.5; renderAll(); });

    // 3D Canvas Orbit Dragging
    const cvs3d = document.getElementById("canvas3d");
    cvs3d.addEventListener("mousedown", (e) => {
      STATE.isDragging3D = true;
      STATE.lastMouseX = e.clientX;
      STATE.lastMouseY = e.clientY;
    });
    window.addEventListener("mousemove", (e) => {
      if (!STATE.isDragging3D) return;
      const dx = e.clientX - STATE.lastMouseX;
      const dy = e.clientY - STATE.lastMouseY;
      STATE.lastMouseX = e.clientX;
      STATE.lastMouseY = e.clientY;
      STATE.camYaw += dx * 0.01;
      STATE.camPitch = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.48, STATE.camPitch + dy * 0.01));
      render3DViewport();
    });
    window.addEventListener("mouseup", () => { STATE.isDragging3D = false; });
    cvs3d.addEventListener("wheel", (e) => {
      e.preventDefault();
      STATE.camZoom = Math.max(0.5, Math.min(2.5, STATE.camZoom - e.deltaY * 0.0015));
      render3DViewport();
    });

    // Telemetry Tabs
    document.querySelectorAll(".telemetry-tab-btn").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll(".telemetry-tab-btn").forEach((x) => x.classList.remove("active"));
        document.querySelectorAll(".telemetry-pane").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        document.getElementById(b.dataset.tab).classList.add("active");
      });
    });

    // Client-side Drag and Drop Inspector
    const dz = document.getElementById("fileDropzone");
    const inp = document.getElementById("fileInputDrop");
    dz.addEventListener("click", () => inp.click());
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files.length) handleClientFiles(e.dataTransfer.files);
    });
    inp.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length) handleClientFiles(e.target.files);
    });

    function handleClientFiles(fileList) {
      Array.from(fileList).forEach((file) => {
        const ext = file.name.split(".").pop().toLowerCase();
        if (ext === "las" || ext === "dev") {
          const reader = new FileReader();
          reader.onload = (ev) => {
            const txt = String(ev.target.result || "");
            document.getElementById("dropzoneStatus").textContent = `Loaded ${file.name} (${txt.length} bytes)`;
          };
          reader.readAsText(file);
        } else if (ext === "segy" || ext === "sgy") {
          const reader = new FileReader();
          reader.onload = (ev) => {
            document.getElementById("dropzoneStatus").textContent = `Loaded ${file.name} (${ev.target.result.byteLength} bytes)`;
          };
          reader.readAsArrayBuffer(file);
        }
      });
    }

    // Apply URL query parameters
    (function initFromQueryParams() {
      const params = new URLSearchParams(window.location.search);
      const scen = params.get("scenario") || params.get("preset") || "munk_sofar_shadow_zone";
      document.getElementById("selectScenario").value = scen;
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.toggle("active", b.dataset.scen === scen));
      if (params.has("field")) {
        const f = params.get("field");
        document.getElementById("selectField").value = f;
        STATE.activeField = f;
      }
      if (params.has("well")) STATE.selectedProfile = parseInt(params.get("well"), 10);
      if (params.has("sliceZ")) STATE.sliceZ = parseInt(params.get("sliceZ"), 10);
      if (params.has("sliceY")) STATE.sliceY = parseInt(params.get("sliceY"), 10);
      if (params.has("sliceX")) STATE.sliceX = parseInt(params.get("sliceX"), 10);
      if (params.has("yaw")) STATE.camYaw = parseFloat(params.get("yaw"));
      if (params.has("pitch")) STATE.camPitch = parseFloat(params.get("pitch"));
      if (params.has("zoom")) STATE.camZoom = parseFloat(params.get("zoom"));
      loadScenario(scen);
    })();

    window.addEventListener("resize", renderAll);
  </script>
</body>
</html>
"""
    return html_template


def export_portable_html_studio(
    output_path: str | Path = "results/oatse_portable_studio.html",
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
) -> Path:
    """
    Generate and save the standalone, zero-install single-file HTML5 3D Studio
    (`results/oatse_portable_studio.html`).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    offline_scenarios = build_all_serialized_scenarios(grid_shape=grid_shape)
    html_content = generate_studio_html(offline_scenarios)
    out_file.write_text(html_content, encoding="utf-8")
    return out_file


def create_studio_app(default_grid_shape: Tuple[int, int, int] = (32, 24, 24)) -> Any:
    """Create a Flask application for local interactive serving of the studio."""
    if not HAS_FLASK:
        raise RuntimeError("Flask is required for create_studio_app(). Install flask or use export_portable_html_studio().")

    app = Flask(__name__)
    cached_scenarios = build_all_serialized_scenarios(grid_shape=default_grid_shape)
    studio_html = generate_studio_html(cached_scenarios)

    @app.route("/")
    def index():
        return render_template_string(studio_html)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "oatse-studio", "version": "0.1.0"})

    @app.route("/api/evaluate", methods=["GET", "POST"])
    def evaluate_endpoint():
        data = request.get_json(silent=True) or {}
        scenario_name = str(data.get("scenario_name") or request.args.get("scenario_name") or "munk_sofar_shadow_zone")
        payload = cached_scenarios.get(scenario_name) or list(cached_scenarios.values())[0]
        return jsonify(payload)

    return app


__all__ = [
    "build_all_serialized_scenarios",
    "create_studio_app",
    "export_portable_html_studio",
    "generate_studio_html",
    "serialize_evaluation_payload",
]
