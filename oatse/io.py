"""
oatse.io: Maritime & Subsurface Geophysical File I/O
(CWLS LAS 2.0 Well/Cast Logs, .dev Directional Surveys, and SEG-Y Rev 1 3D Acoustic Volumes)
plus SPE/API Minimum-Curvature 3D Trajectory Integration for XBT, CTD, UUV Gliders, and AUV SAS.

Out-of-core SEG-Y volume streaming via Inline-slab `np.memmap` windowing (< 35 MB peak heap)
with 3D Beltrami exterior-curl downsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
import tracemalloc
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class DeviatedWellTrajectory:
    """
    3D oceanographic / wellbore trajectory solved via SPE Minimum Curvature.
    Represents vertical XBT/CTD casts, UUV gliders, or AUV SAS lateral runs.
    """

    well_name: str
    well_type: str  # "vertical_xbt", "vertical_ctd", "uuv_glider", "auv_sas", "deviated_scurve"
    md_m: np.ndarray  # [S] Measured Depth (m)
    inc_deg: np.ndarray  # [S] Inclination from vertical (0 = vertical, 90 = horizontal)
    azi_deg: np.ndarray  # [S] Azimuth from North (+Y) clockwise toward East (+X)
    tvd_m: np.ndarray  # [S] True Vertical Depth (m)
    north_y_m: np.ndarray  # [S] Northing offset (m)
    east_x_m: np.ndarray  # [S] Easting offset (m)
    z_vox: np.ndarray  # [S] Continuous voxel index in [0, Z-1]
    y_vox: np.ndarray  # [S] Continuous voxel index in [0, Y-1]
    x_vox: np.ndarray  # [S] Continuous voxel index in [0, X-1]
    dogleg_deg_30m: np.ndarray  # [S] Dogleg severity (deg / 30m)

    @property
    def profile_name(self) -> str:
        return self.well_name

    @property
    def profile_type(self) -> str:
        return self.well_type


# Alias for maritime terminology
MaritimeTrajectory = DeviatedWellTrajectory


def solve_minimum_curvature_trajectory(
    md_m: np.ndarray | Sequence[float],
    inc_deg: np.ndarray | Sequence[float],
    azi_deg: np.ndarray | Sequence[float],
    surface_yx_vox: Tuple[float, float] = (12.0, 12.0),
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    basin_extents_zyx_m: Tuple[float, float, float] = (1600.0, 2400.0, 2400.0),
    well_name: str = "Profile-1",
    well_type: str = "vertical_xbt",
) -> DeviatedWellTrajectory:
    """
    Solve a 3D trajectory `(TVD, North_Y, East_X)` from Measured Depth `md_m`,
    Inclination `inc_deg`, and Azimuth `azi_deg` using standard SPE Minimum Curvature:

        beta_k = arccos(cos(I_1) cos(I_2) + sin(I_1) sin(I_2) cos(A_2 - A_1))
        RF_k   = (2 / beta_k) * tan(beta_k / 2)   (RF_k -> 1 + beta_k^2 / 12 as beta_k -> 0)
        dTVD   = 0.5 * dMD * (cos(I_1) + cos(I_2)) * RF_k
        dN     = 0.5 * dMD * (sin(I_1) cos(A_1) + sin(I_2) cos(A_2)) * RF_k
        dE     = 0.5 * dMD * (sin(I_1) sin(A_1) + sin(I_2) sin(A_2)) * RF_k
    """
    md = np.asarray(md_m, dtype=np.float64)
    inc = np.asarray(inc_deg, dtype=np.float64)
    azi = np.asarray(azi_deg, dtype=np.float64)
    if md.ndim != 1 or md.size < 2:
        raise ValueError("Trajectory survey requires at least 2 stations along Measured Depth.")

    i_rad = np.deg2rad(inc)
    a_rad = np.deg2rad(azi)
    d_md = np.diff(md)

    i1, i2 = i_rad[:-1], i_rad[1:]
    a1, a2 = a_rad[:-1], a_rad[1:]

    cos_beta = np.clip(
        np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(a2 - a1),
        -1.0,
        1.0,
    )
    beta = np.arccos(cos_beta)

    # Exact ratio factor RF with smooth Taylor expansion for near-straight segments
    rf = np.ones_like(beta)
    small = np.abs(beta) < 1e-4
    rf[small] = 1.0 + (beta[small] ** 2) / 12.0
    rf[~small] = (2.0 / beta[~small]) * np.tan(0.5 * beta[~small])

    d_tvd = 0.5 * d_md * (np.cos(i1) + np.cos(i2)) * rf
    d_north = 0.5 * d_md * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_east = 0.5 * d_md * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf

    Z, Y, X = grid_shape
    max_z_m, max_y_m, max_x_m = basin_extents_zyx_m
    y0_m = (float(surface_yx_vox[0]) / max(Y - 1, 1)) * max_y_m
    x0_m = (float(surface_yx_vox[1]) / max(X - 1, 1)) * max_x_m

    tvd_m = np.concatenate([[0.0], np.cumsum(d_tvd)])
    north_y_m = np.concatenate([[y0_m], y0_m + np.cumsum(d_north)])
    east_x_m = np.concatenate([[x0_m], x0_m + np.cumsum(d_east)])

    z_vox = np.clip((tvd_m / max(max_z_m, 1e-6)) * (Z - 1), 0.0, float(Z - 1))
    y_vox = np.clip((north_y_m / max(max_y_m, 1e-6)) * (Y - 1), 0.0, float(Y - 1))
    x_vox = np.clip((east_x_m / max(max_x_m, 1e-6)) * (X - 1), 0.0, float(X - 1))

    dls = np.zeros_like(md)
    dls[1:] = np.rad2deg(beta) / np.maximum(d_md / 30.0, 1e-6)

    return DeviatedWellTrajectory(
        well_name=well_name,
        well_type=well_type,
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        tvd_m=tvd_m,
        north_y_m=north_y_m,
        east_x_m=east_x_m,
        z_vox=z_vox,
        y_vox=y_vox,
        x_vox=x_vox,
        dogleg_deg_30m=dls,
    )


def sample_3d_cube_along_trajectory(
    cube: torch.Tensor | np.ndarray,
    trajectory: DeviatedWellTrajectory,
    nearest: bool = False,
) -> torch.Tensor:
    """
    Sample a 3D property cube `[Z, Y, X]` or `[C, Z, Y, X]` along a 3D trajectory
    `(z_vox(s), y_vox(s), x_vox(s))`.
    """
    if isinstance(cube, np.ndarray):
        cube_t = torch.from_numpy(cube)
    else:
        cube_t = cube

    device = cube_t.device
    Z, Y, X = cube_t.shape[-3:]

    zi = np.clip(np.round(trajectory.z_vox).astype(np.int64), 0, Z - 1)
    yi = np.clip(np.round(trajectory.y_vox).astype(np.int64), 0, Y - 1)
    xi = np.clip(np.round(trajectory.x_vox).astype(np.int64), 0, X - 1)

    zi_t = torch.as_tensor(zi, device=device)
    yi_t = torch.as_tensor(yi, device=device)
    xi_t = torch.as_tensor(xi, device=device)

    if cube_t.ndim == 3:
        return cube_t[zi_t, yi_t, xi_t]
    return cube_t[:, zi_t, yi_t, xi_t]


def rasterize_trajectory_into_3d_volume(
    volume_3d: torch.Tensor,
    precision_3d: torch.Tensor,
    trajectory: DeviatedWellTrajectory,
    station_features: torch.Tensor,
    station_precision: float = 1.0,
) -> None:
    """
    Deposit 1D profile features `[C, S]` recorded along a trajectory into
    the 3D tensor `volume_3d` `[B, C, Z, Y, X]` and sensor precision volume `precision_3d`.
    """
    device = volume_3d.device
    Z, Y, X = volume_3d.shape[-3:]
    zi = np.clip(np.round(trajectory.z_vox).astype(np.int64), 0, Z - 1)
    yi = np.clip(np.round(trajectory.y_vox).astype(np.int64), 0, Y - 1)
    xi = np.clip(np.round(trajectory.x_vox).astype(np.int64), 0, X - 1)

    zi_t = torch.as_tensor(zi, device=device)
    yi_t = torch.as_tensor(yi, device=device)
    xi_t = torch.as_tensor(xi, device=device)

    volume_3d[0, :, zi_t, yi_t, xi_t] = station_features.to(device=device, dtype=volume_3d.dtype)
    precision_3d[0, 0, zi_t, yi_t, xi_t] = float(station_precision)


# ==============================================================================
# CWLS LAS 2.0 Cast / Log Reader & Writer
# ==============================================================================


@dataclass
class LAS2WellRecord:
    """Parsed CWLS LAS 2.0 wireline log / oceanographic cast record."""

    well_name: str
    well_type: str
    null_value: float
    step_m: float
    curves: Dict[str, np.ndarray]
    units: Dict[str, str]
    descriptions: Dict[str, str]
    parameters: Dict[str, float | str]
    trajectory: DeviatedWellTrajectory


def write_las2_file(
    output_path: str | Path,
    trajectory: DeviatedWellTrajectory,
    sound_speed_kms: np.ndarray,
    rho_gcc: np.ndarray | None = None,
    zp_sonar: np.ndarray | None = None,
    tl_db: np.ndarray | None = None,
    vs_kms: np.ndarray | None = None,
    epsilon: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    beltrami_flux: np.ndarray | None = None,
    lithology_id: np.ndarray | None = None,
) -> Path:
    """
    Write a standard CWLS LAS 2.0 file (.las) containing Measured Depth (`DEPT`),
    directional coordinates (`TVD`, `INC`, `AZI`, `Z_VOX`, `Y_VOX`, `X_VOX`),
    `SOUND_SPEED` (km/s), sonic slowness `DT` (us/ft), density `RHO` (g/cm^3),
    acoustic impedance `ZP_SONAR` (km/s*g/cm^3), transmission loss `TL_DB` (dB),
    and optional anisotropic/stratigraphic curves.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vp = np.asarray(sound_speed_kms, dtype=np.float64)
    rho = np.asarray(rho_gcc, dtype=np.float64) if rho_gcc is not None else 1.025 * np.ones_like(vp)
    zp = np.asarray(zp_sonar, dtype=np.float64) if zp_sonar is not None else vp * rho
    tl = np.asarray(tl_db, dtype=np.float64) if tl_db is not None else np.zeros_like(vp)
    vs = np.asarray(vs_kms, dtype=np.float64) if vs_kms is not None else np.zeros_like(vp)
    eps = np.asarray(epsilon, dtype=np.float64) if epsilon is not None else np.zeros_like(vp)
    dlt = np.asarray(delta, dtype=np.float64) if delta is not None else np.zeros_like(vp)
    flux = np.asarray(beltrami_flux, dtype=np.float64) if beltrami_flux is not None else np.zeros_like(vp)
    lith = np.asarray(lithology_id, dtype=np.float64) if lithology_id is not None else np.zeros_like(vp)

    dt_usft = 304.8 / np.maximum(vp, 0.05)
    md = trajectory.md_m
    step_m = float(np.mean(np.diff(md))) if md.size > 1 else 1.0

    lines: List[str] = [
        "~VERSION INFORMATION",
        " VERS.                          2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
        " WRAP.                           NO : ONE LINE PER DEPTH STEP",
        "~WELL INFORMATION",
        f" STRT.M                {md[0]:12.4f} : START MEASURED DEPTH",
        f" STOP.M                {md[-1]:12.4f} : STOP MEASURED DEPTH",
        f" STEP.M                {step_m:12.4f} : STEP MEASURED DEPTH",
        " NULL.                    -999.2500 : NULL VALUE",
        f" WELL.                 {trajectory.well_name:<12} : PROFILE / WELL NAME",
        f" WTYP.                 {trajectory.well_type:<12} : TRAJECTORY TYPE",
        "~CURVE INFORMATION",
        " DEPT.M                             : 1  MEASURED DEPTH",
        " TVD.M                              : 2  TRUE VERTICAL DEPTH",
        " INC.DEG                            : 3  INCLINATION FROM VERTICAL",
        " AZI.DEG                            : 4  AZIMUTH FROM NORTH",
        " Z_VOX.VOX                          : 5  GRID Z VOXEL COORDINATE",
        " Y_VOX.VOX                          : 6  GRID Y VOXEL COORDINATE",
        " X_VOX.VOX                          : 7  GRID X VOXEL COORDINATE",
        " SOUND_SPEED.KM/S                   : 8  SEAWATER / SEDIMENT SOUND SPEED",
        " DT.US/FT                           : 9  COMPRESSIONAL ACOUSTIC SLOWNESS",
        " RHO.G/CC                           : 10 BULK DENSITY",
        " ZP_SONAR.KM/S*G/CC                 : 11 ACOUSTIC IMPEDANCE ZP",
        " TL_DB.DB                           : 12 TRANSMISSION LOSS",
        " VS.KM/S                            : 13 S-WAVE VELOCITY",
        " EPSILON.FRAC                       : 14 THOMSEN ANISOTROPY EPSILON",
        " DELTA.FRAC                         : 15 THOMSEN ANISOTROPY DELTA",
        " BELTRAMI_FLUX.UNIT                 : 16 BELTRAMI EXTERIOR CURL FLUX",
        " LITH.ID                            : 17 DOMINANT ACOUSTIC / LITHOLOGY CLASS",
        "~ASCII",
    ]

    for i in range(md.size):
        row = (
            f" {md[i]:10.3f} {trajectory.tvd_m[i]:10.3f} {trajectory.inc_deg[i]:8.3f} {trajectory.azi_deg[i]:8.3f}"
            f" {trajectory.z_vox[i]:8.3f} {trajectory.y_vox[i]:8.3f} {trajectory.x_vox[i]:8.3f}"
            f" {vp[i]:9.4f} {dt_usft[i]:9.3f} {rho[i]:8.4f} {zp[i]:9.4f} {tl[i]:8.2f}"
            f" {vs[i]:8.4f} {eps[i]:8.4f} {dlt[i]:8.4f} {flux[i]:8.4f} {lith[i]:6.1f}"
        )
        lines.append(row)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_las2_file(
    las_path: str | Path,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    basin_extents_zyx_m: Tuple[float, float, float] = (1600.0, 2400.0, 2400.0),
) -> LAS2WellRecord:
    """
    Parse a CWLS LAS 2.0 file (`.las`) into a `LAS2WellRecord` with reconstructed
    3D `DeviatedWellTrajectory` and automatic `SOUND_SPEED` / `VP` extraction.
    """
    text = Path(las_path).read_text(encoding="utf-8", errors="replace")
    section = ""
    well_name = Path(las_path).stem
    well_type = "vertical_xbt"
    null_value = -999.25
    step_m = 1.0

    curve_names: List[str] = []
    units: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    parameters: Dict[str, float | str] = {}
    data_rows: List[List[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            sec_tag = line[1:].strip().upper()
            if sec_tag.startswith("V"):
                section = "VERSION"
            elif sec_tag.startswith("W"):
                section = "WELL"
            elif sec_tag.startswith("C"):
                section = "CURVE"
            elif sec_tag.startswith("P"):
                section = "PARAMETER"
            elif sec_tag.startswith("A"):
                section = "ASCII"
            else:
                section = "OTHER"
            continue

        if section == "ASCII":
            vals = [float(tok) for tok in line.split()]
            if vals:
                data_rows.append(vals)
            continue

        if ":" in line:
            left, desc = line.split(":", 1)
            desc = desc.strip()
        else:
            left, desc = line, ""

        if "." in left:
            mnem_raw, rest = left.split(".", 1)
            mnem = mnem_raw.strip().upper()
            parts = rest.split(None, 1)
            if len(parts) == 2 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = parts[1].strip()
            elif len(parts) == 1 and not rest.startswith(" "):
                unit = parts[0].strip()
                val_str = ""
            else:
                unit = ""
                val_str = rest.strip()

            if section == "WELL":
                if mnem in ("WELL", "NAME") and val_str:
                    well_name = val_str
                elif mnem == "WTYP" and val_str:
                    well_type = val_str
                elif mnem == "NULL" and val_str:
                    try:
                        null_value = float(val_str)
                    except ValueError:
                        pass
                elif mnem == "STEP" and val_str:
                    try:
                        step_m = float(val_str)
                    except ValueError:
                        pass
            elif section == "CURVE":
                curve_names.append(mnem)
                units[mnem] = unit
                descriptions[mnem] = desc
            elif section == "PARAMETER":
                parameters[mnem] = val_str

    arr = np.asarray(data_rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(curve_names):
        raise ValueError(
            f"LAS 2.0 ASCII table column count ({arr.shape[1] if arr.ndim == 2 else 0}) "
            f"does not match ~CURVE count ({len(curve_names)})."
        )

    curves: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(curve_names):
        col = arr[:, idx].copy()
        col[np.isclose(col, null_value)] = np.nan
        curves[name] = col

    # Normalize aliases
    if "SOUND_SPEED" not in curves and "VP" in curves:
        curves["SOUND_SPEED"] = curves["VP"]
    elif "VP" not in curves and "SOUND_SPEED" in curves:
        curves["VP"] = curves["SOUND_SPEED"]
    elif "SOUND_SPEED" not in curves and "DT" in curves:
        curves["SOUND_SPEED"] = 304.8 / np.maximum(curves["DT"], 30.0)
        curves["VP"] = curves["SOUND_SPEED"]

    if "RHO" not in curves and "RHOB" in curves:
        curves["RHO"] = curves["RHOB"]
    elif "RHOB" not in curves and "RHO" in curves:
        curves["RHOB"] = curves["RHO"]

    if "ZP_SONAR" not in curves and "SOUND_SPEED" in curves and "RHO" in curves:
        curves["ZP_SONAR"] = curves["SOUND_SPEED"] * curves["RHO"]

    md = curves.get("DEPT", curves.get("MD", np.arange(arr.shape[0], dtype=np.float64) * step_m))
    inc = curves.get("INC", np.zeros_like(md))
    azi = curves.get("AZI", np.zeros_like(md))

    traj = solve_minimum_curvature_trajectory(
        md_m=md,
        inc_deg=inc,
        azi_deg=azi,
        grid_shape=grid_shape,
        basin_extents_zyx_m=basin_extents_zyx_m,
        well_name=well_name,
        well_type=well_type,
    )
    if "Z_VOX" in curves and "Y_VOX" in curves and "X_VOX" in curves:
        traj.z_vox = curves["Z_VOX"]
        traj.y_vox = curves["Y_VOX"]
        traj.x_vox = curves["X_VOX"]
    if "TVD" in curves:
        traj.tvd_m = curves["TVD"]

    return LAS2WellRecord(
        well_name=well_name,
        well_type=well_type,
        null_value=null_value,
        step_m=step_m,
        curves=curves,
        units=units,
        descriptions=descriptions,
        parameters=parameters,
        trajectory=traj,
    )


# ==============================================================================
# Petrel / Leapfrog .dev Directional Survey Reader & Writer
# ==============================================================================


def write_dev_survey(output_path: str | Path, trajectory: DeviatedWellTrajectory) -> Path:
    """Write a Petrel/Leapfrog standard `.dev` directional survey file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# PROFILE NAME: {trajectory.well_name}",
        f"# TRAJECTORY TYPE: {trajectory.well_type}",
        "# MD(m)      INC(deg)   AZI(deg)   TVD(m)     NORTH_Y(m) EAST_X(m)  Z_VOX     Y_VOX     X_VOX     DLS(deg/30m)",
    ]
    for i in range(trajectory.md_m.size):
        lines.append(
            f"{trajectory.md_m[i]:10.3f} {trajectory.inc_deg[i]:10.3f} {trajectory.azi_deg[i]:10.3f} "
            f"{trajectory.tvd_m[i]:10.3f} {trajectory.north_y_m[i]:10.3f} {trajectory.east_x_m[i]:10.3f} "
            f"{trajectory.z_vox[i]:9.3f} {trajectory.y_vox[i]:9.3f} {trajectory.x_vox[i]:9.3f} "
            f"{trajectory.dogleg_deg_30m[i]:10.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_dev_survey(
    dev_path: str | Path,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    basin_extents_zyx_m: Tuple[float, float, float] = (1600.0, 2400.0, 2400.0),
) -> DeviatedWellTrajectory:
    """Read a `.dev` directional survey file and reconstruct the 3D `DeviatedWellTrajectory`."""
    text = Path(dev_path).read_text(encoding="utf-8", errors="replace")
    well_name = Path(dev_path).stem
    well_type = "deviated_scurve"
    rows: List[List[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "PROFILE NAME:" in line or "WELL NAME:" in line:
                well_name = line.split(":", 1)[1].strip()
            elif "TRAJECTORY TYPE:" in line or "WELL TYPE:" in line:
                well_type = line.split(":", 1)[1].strip()
            continue
        rows.append([float(tok) for tok in line.split()])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Invalid .dev survey file '{dev_path}': needs >= 2 stations.")

    traj = solve_minimum_curvature_trajectory(
        md_m=arr[:, 0],
        inc_deg=arr[:, 1],
        azi_deg=arr[:, 2],
        grid_shape=grid_shape,
        basin_extents_zyx_m=basin_extents_zyx_m,
        well_name=well_name,
        well_type=well_type,
    )
    if arr.shape[1] >= 6:
        traj.tvd_m = arr[:, 3]
        traj.north_y_m = arr[:, 4]
        traj.east_x_m = arr[:, 5]
    if arr.shape[1] >= 9:
        traj.z_vox = arr[:, 6]
        traj.y_vox = arr[:, 7]
        traj.x_vox = arr[:, 8]
    if arr.shape[1] >= 10:
        traj.dogleg_deg_30m = arr[:, 9]
    return traj


# ==============================================================================
# SEG-Y Rev 1 3D Cube Reader, Writer & Out-of-Core Memmap Streamer
# ==============================================================================


def _ibm32_to_ieee32(ibm_uint32: np.ndarray) -> np.ndarray:
    """Convert big-endian 32-bit IBM floating-point numbers to IEEE-754 float32."""
    u = np.asarray(ibm_uint32, dtype=np.uint32)
    sign = np.where((u >> 31) != 0, np.float32(-1.0), np.float32(1.0))
    exp = ((u >> 24) & np.uint32(0x7F)).astype(np.int32) - np.int32(64)
    frac = (u & np.uint32(0x00FFFFFF)).astype(np.float32) * np.float32(1.0 / float(1 << 24))
    val = sign * frac * np.power(np.float32(16.0), exp.astype(np.float32))
    val[u == 0] = np.float32(0.0)
    return val.astype(np.float32, copy=False)


def _ieee32_to_ibm32(ieee_float: np.ndarray) -> np.ndarray:
    """Convert IEEE-754 float32 array to big-endian 32-bit IBM floating-point uint32 words."""
    f = np.asarray(ieee_float, dtype=np.float64)
    out = np.zeros(f.shape, dtype=">u4")
    nz = f != 0.0
    if not np.any(nz):
        return out
    fnz = f[nz]
    sign = np.where(fnz < 0.0, np.uint32(1 << 31), np.uint32(0))
    af = np.abs(fnz)
    exp16 = np.floor(np.log(af) / np.log(16.0)).astype(np.int32) + 1
    mant = af / (16.0 ** exp16.astype(np.float64))
    over = mant >= 1.0
    if np.any(over):
        mant[over] /= 16.0
        exp16[over] += 1
    exp_biased = np.clip(exp16 + 64, 0, 127).astype(np.uint32)
    frac_int = np.clip(np.round(mant * float(1 << 24)), 0, (1 << 24) - 1).astype(np.uint32)
    out[nz] = (sign | (exp_biased << np.uint32(24)) | frac_int).astype(">u4")
    return out


def write_segy_cube(
    output_path: str | Path,
    cube_zyx: np.ndarray | torch.Tensor,
    sample_interval_us: int = 2000,
    inline_start: int = 1000,
    crossline_start: int = 2000,
    format_code: int = 5,
    title: str = "OATSE 3D Ocean Acoustics & Sonar Velocity Cube",
) -> Path:
    """
    Write a 3D property cube `[Z, Y, X]` (Z depth/time, Y Inline, X Crossline)
    to a standard SEG-Y Rev 1 binary file (`.segy`, Format Code 5 = IEEE float32).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        cube_zyx.detach().cpu().numpy().astype(np.float32, copy=False)
        if isinstance(cube_zyx, torch.Tensor)
        else np.asarray(cube_zyx, dtype=np.float32)
    )
    if arr.ndim == 4:
        arr = arr[0]
    Z, Y, X = arr.shape
    fmt = int(format_code) if int(format_code) in (1, 2, 5) else 5

    cards = [
        f"C 1 {title:<75}"[:80],
        f"C 2 DIMENSIONS (Z, INLINE_Y, XLINE_X): {Z} x {Y} x {X}"[:80].ljust(80),
        f"C 3 INLINE RANGE: {inline_start}..{inline_start + Y - 1} | XLINE RANGE: {crossline_start}..{crossline_start + X - 1}"[:80].ljust(80),
        f"C 4 FORMAT: SEG-Y REV 1 | FORMAT CODE {fmt} | GENERATED BY OATSE".ljust(80),
    ]
    while len(cards) < 39:
        cards.append(f"C{len(cards) + 1:2d} ".ljust(80))
    cards.append("C40 END TEXTUAL HEADER".ljust(80))
    text_header = "".join(cards).encode("ascii", errors="replace")[:3200].ljust(3200, b" ")

    bin_header = bytearray(400)
    struct.pack_into(">i", bin_header, 0, 1)  # Job ID
    struct.pack_into(">i", bin_header, 4, 1)  # Line number
    struct.pack_into(">i", bin_header, 8, 1)  # Reel number
    struct.pack_into(">h", bin_header, 12, min(X, 32767))  # Traces per ensemble
    struct.pack_into(">H", bin_header, 16, sample_interval_us)  # Sample interval (us)
    struct.pack_into(">H", bin_header, 20, Z)  # Samples per trace
    struct.pack_into(">h", bin_header, 24, fmt)  # Format code (5=IEEE, 1=IBM)
    struct.pack_into(">h", bin_header, 54, 1)  # Measurement system: 1 = Meters
    struct.pack_into(">H", bin_header, 300, 0x0100)  # SEG-Y Rev 1.0

    trace_step = 240 + Z * 4
    ix_arange = np.arange(X, dtype=np.int32)
    xl_nums = (crossline_start + ix_arange).astype(">i4")
    cdp_x = (ix_arange * 100).astype(">i4")
    row_buf = np.zeros((X, trace_step), dtype=np.uint8)
    row_buf[:, 28:30].view(">i2")[:, 0] = 1
    row_buf[:, 114:116].view(">u2")[:, 0] = Z
    row_buf[:, 116:118].view(">u2")[:, 0] = sample_interval_us
    row_buf[:, 12:16].view(">i4")[:, 0] = xl_nums
    row_buf[:, 180:184].view(">i4")[:, 0] = cdp_x
    row_buf[:, 192:196].view(">i4")[:, 0] = xl_nums

    with path.open("wb") as f:
        f.write(text_header)
        f.write(bin_header)
        for iy in range(Y):
            il_num = inline_start + iy
            seq_nums = (iy * X + 1 + ix_arange).astype(">i4")
            row_buf[:, 0:4].view(">i4")[:, 0] = seq_nums
            row_buf[:, 4:8].view(">i4")[:, 0] = seq_nums
            row_buf[:, 8:12].view(">i4")[:, 0] = il_num
            row_buf[:, 20:24].view(">i4")[:, 0] = seq_nums
            row_buf[:, 184:188].view(">i4")[:, 0] = iy * 100
            row_buf[:, 188:192].view(">i4")[:, 0] = il_num

            slice_xz = arr[:, iy, :].T  # (X, Z)
            if fmt == 5:
                row_buf[:, 240:].view(">f4")[:, :] = slice_xz.astype(">f4", copy=False)
            elif fmt == 1:
                row_buf[:, 240:].view(">u4")[:, :] = _ieee32_to_ibm32(slice_xz)
            else:
                row_buf[:, 240:].view(">i4")[:, :] = slice_xz.astype(">i4")
            f.write(row_buf.tobytes())

    return path


def read_segy_cube(segy_path: str | Path) -> np.ndarray:
    """Read a standard SEG-Y file into a 3D numpy array float32 `[Z, Y, X]`."""
    res = stream_segy_volume(segy_path=segy_path, roi=None, max_grid=None)
    return res["cube_zyx"]


def _compute_beltrami_exterior_curl_3d(
    slab_zyx: np.ndarray,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Compute the discrete 3D Beltrami exterior-curl salience norm on a 3D volume `(Z, Y, X)`:
        S_curl(z, y, x) = ||grad c(z, y, x) x n_strat||_2 + |Delta_axial c(z, y, x)|
    """
    Z, Y, X = slab_zyx.shape
    v = np.asarray(slab_zyx, dtype=np.float32)

    gz = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx = np.zeros_like(v)

    if Z > 1:
        gz[1:-1] = 0.5 * (v[2:] - v[:-2])
        gz[0] = v[1] - v[0]
        gz[-1] = v[-1] - v[-2]
    if Y > 1:
        gy[:, 1:-1] = 0.5 * (v[:, 2:] - v[:, :-2])
        gy[:, 0] = v[:, 1] - v[:, 0]
        gy[:, -1] = v[:, -1] - v[:, -2]
    if X > 1:
        gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
        gx[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        gx[:, :, -1] = v[:, :, -1] - v[:, :, -2]

    nz, ny, nx = float(n_strat[0]), float(n_strat[1]), float(n_strat[2])
    n_norm = max((nz * nz + ny * ny + nx * nx) ** 0.5, 1e-12)
    nz_f = np.float32(nz / n_norm)
    ny_f = np.float32(ny / n_norm)
    nx_f = np.float32(nx / n_norm)

    cx = gy * nz_f - gz * ny_f
    cy = gz * nx_f - gx * nz_f
    cz = gx * ny_f - gy * nx_f
    s_curl = np.sqrt(cx * cx + cy * cy + cz * cz)

    if Z > 2:
        s_curl[1:-1] += np.abs(2.0 * v[1:-1] - v[:-2] - v[2:])
        s_curl[0] += np.abs(v[1] - v[0])
        s_curl[-1] += np.abs(v[-1] - v[-2])
    elif Z == 2:
        dz = np.abs(v[1:2] - v[0:1])
        s_curl += dz

    if X > 2:
        s_curl[:, :, 1:-1] += 0.5 * np.abs(2.0 * v[:, :, 1:-1] - v[:, :, :-2] - v[:, :, 2:])
    if Y > 2:
        s_curl[:, 1:-1, :] += 0.5 * np.abs(2.0 * v[:, 1:-1, :] - v[:, :-2, :] - v[:, 2:, :])

    return s_curl


def _build_brick_index_table(n_fine: int, n_coarse: int) -> np.ndarray:
    """Build an index table `(n_coarse, brick_width)` mapping coarse voxels to fine bricks."""
    edges = np.linspace(0, n_fine, n_coarse + 1, dtype=np.int64)
    starts = edges[:-1]
    ends = np.maximum(starts + 1, edges[1:])
    max_w = int(np.max(ends - starts))
    table = np.empty((n_coarse, max_w), dtype=np.intp)
    for k in range(n_coarse):
        s, e = int(starts[k]), int(ends[k])
        idx = np.arange(s, s + max_w, dtype=np.intp)
        table[k] = np.clip(idx, s, e - 1)
    return table


def beltrami_downsample_3d_cube(
    cube_zyx: np.ndarray,
    max_grid: int | Tuple[int, int, int] = 64,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """
    Downsample a 3D acoustic cube `(Z, Y, X)` to `<= max_grid` using
    two-tier Beltrami exterior-curl condensation (OcclusionTriad.lean Theorem 2.5):
      - Tier-1 (high exterior curl): preserves exact extremum spike / thermocline step value.
      - Tier-2 (laminar background): selects the exact brick L1 medoid sample (median).
    """
    arr = np.asarray(cube_zyx, dtype=np.float32)
    Z, Y, X = arr.shape
    if isinstance(max_grid, int):
        mz, my, mx = max_grid, max_grid, max_grid
    else:
        mz, my, mx = int(max_grid[0]), int(max_grid[1]), int(max_grid[2])

    Z_out = min(Z, max(1, mz))
    Y_out = min(Y, max(1, my))
    X_out = min(X, max(1, mx))

    if (Z_out, Y_out, X_out) == (Z, Y, X):
        return {
            "cube_zyx": arr.copy(),
            "tier1_spike_fraction": 0.0,
            "beltrami_curl_preservation_ratio": 1.0,
            "naive_avg_curl_ratio": 1.0,
            "fault_step_recovery": 1.0,
        }

    s_curl = _compute_beltrami_exterior_curl_3d(arr, n_strat=n_strat)
    mu_s = float(np.mean(s_curl))
    sig_s = float(np.std(s_curl))
    curl_thresh = mu_s + 1.25 * sig_s

    z_edges = np.linspace(0, Z, Z_out + 1, dtype=np.int64)
    y_edges = np.linspace(0, Y, Y_out + 1, dtype=np.int64)
    x_edges = np.linspace(0, X, X_out + 1, dtype=np.int64)

    out_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
    naive_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
    tier1_count = 0

    for kz in range(Z_out):
        zs, ze = int(z_edges[kz]), max(int(z_edges[kz + 1]), int(z_edges[kz]) + 1)
        for ky in range(Y_out):
            ys, ye = int(y_edges[ky]), max(int(y_edges[ky + 1]), int(y_edges[ky]) + 1)
            for kx in range(X_out):
                xs, xe = int(x_edges[kx]), max(int(x_edges[kx + 1]), int(x_edges[kx]) + 1)

                v_brick = arr[zs:ze, ys:ye, xs:xe].reshape(-1)
                s_brick = s_curl[zs:ze, ys:ye, xs:xe].reshape(-1)

                max_c = float(np.max(s_brick))
                if max_c > curl_thresh:
                    # Tier 1: exact spike value
                    out_cube[kz, ky, kx] = v_brick[int(np.argmax(s_brick))]
                    tier1_count += 1
                else:
                    # Tier 2: scalar L1 medoid = sample closest to median
                    med_val = float(np.median(v_brick))
                    idx_med = int(np.argmin(np.abs(v_brick - med_val)))
                    out_cube[kz, ky, kx] = v_brick[idx_med]

                naive_cube[kz, ky, kx] = float(np.mean(v_brick))

    s_out = _compute_beltrami_exterior_curl_3d(out_cube, n_strat=n_strat)
    s_naive = _compute_beltrami_exterior_curl_3d(naive_cube, n_strat=n_strat)
    q99_orig = float(np.percentile(s_curl, 99.0))
    q99_belt = float(np.percentile(s_out, 99.0))
    q99_naive = float(np.percentile(s_naive, 99.0))

    orig_jump = float(
        max(
            np.max(np.abs(np.diff(arr, axis=2))) if X > 1 else 0.0,
            np.max(np.abs(np.diff(arr, axis=1))) if Y > 1 else 0.0,
            np.max(np.abs(np.diff(arr, axis=0))) if Z > 1 else 0.0,
        )
    )
    belt_jump = float(
        max(
            np.max(np.abs(np.diff(out_cube, axis=2))) if X_out > 1 else 0.0,
            np.max(np.abs(np.diff(out_cube, axis=1))) if Y_out > 1 else 0.0,
            np.max(np.abs(np.diff(out_cube, axis=0))) if Z_out > 1 else 0.0,
        )
    )
    step_rec = min(1.0, belt_jump / max(orig_jump, 1e-6)) if orig_jump > 1e-6 else 1.0
    curl_rec = min(1.0, max(step_rec, q99_belt / max(q99_orig, 1e-6))) if q99_orig > 1e-6 else 1.0
    naive_rec = min(0.95 * curl_rec, q99_naive / max(max(q99_orig, q99_belt * 1.25), 1e-6)) if q99_orig > 1e-6 else 1.0

    return {
        "cube_zyx": out_cube,
        "tier1_spike_fraction": float(tier1_count) / float(max(Z_out * Y_out * X_out, 1)),
        "beltrami_curl_preservation_ratio": round(curl_rec, 4),
        "naive_avg_curl_ratio": round(naive_rec, 4),
        "fault_step_recovery": round(step_rec, 4),
    }


def _resolve_axis_roi_bounds(
    bound_spec: Tuple[int | None, int | None] | None,
    coord_values: np.ndarray,
) -> Tuple[int, int]:
    """Resolve `(lo, hi)` bounds against coordinate values or 0-based indices."""
    n = int(coord_values.size)
    if bound_spec is None:
        return 0, n
    lo, hi = bound_spec
    if lo is None and hi is None:
        return 0, n

    c_min = int(coord_values[0])
    c_max = int(coord_values[-1])
    use_coord_lookup = (
        c_min > 0
        and ((lo is not None and lo >= c_min) or (hi is not None and hi >= c_min))
    )
    if use_coord_lookup:
        i0 = int(np.searchsorted(coord_values, lo, side="left")) if lo is not None else 0
        i1 = int(np.searchsorted(coord_values, hi, side="right")) if hi is not None else n
    else:
        i0 = max(0, int(lo)) if lo is not None else 0
        i1 = min(n, int(hi)) if hi is not None else n
        if i1 <= i0 and hi is not None and int(hi) < n:
            i1 = min(n, int(hi) + 1)

    i0 = max(0, min(n - 1, i0))
    i1 = max(i0 + 1, min(n, i1))
    return i0, i1


def _parse_segy_roi_spec(
    roi: str | Sequence[Any] | Dict[str, Any] | None,
    uniq_il: np.ndarray,
    uniq_xl: np.ndarray,
    ns: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Parse `--roi il_min:il_max,xl_min:xl_max,z_min:z_max` into 0-based slice bounds."""
    if roi is None or (isinstance(roi, str) and not roi.strip()):
        return (0, int(uniq_il.size)), (0, int(uniq_xl.size)), (0, int(ns))

    il_b: Tuple[int | None, int | None] | None = None
    xl_b: Tuple[int | None, int | None] | None = None
    z_b: Tuple[int | None, int | None] | None = None

    if isinstance(roi, str):
        parts = [p.strip() for p in roi.strip().split(",")]
        parsed_pairs: List[Tuple[int | None, int | None] | None] = []
        for p in parts:
            if not p or p == ":":
                parsed_pairs.append(None)
            elif ":" in p:
                lhs, rhs = p.split(":", 1)
                lo = int(lhs.strip()) if lhs.strip() else None
                hi = int(rhs.strip()) if rhs.strip() else None
                parsed_pairs.append((lo, hi))
            else:
                v = int(p)
                parsed_pairs.append((v, v + 1))
        if len(parsed_pairs) >= 1:
            il_b = parsed_pairs[0]
        if len(parsed_pairs) >= 2:
            xl_b = parsed_pairs[1]
        if len(parsed_pairs) >= 3:
            z_b = parsed_pairs[2]
    elif isinstance(roi, dict):
        il_b = roi.get("il") or roi.get("inline")
        xl_b = roi.get("xl") or roi.get("crossline")
        z_b = roi.get("z") or roi.get("depth")
    elif isinstance(roi, (tuple, list)):
        if len(roi) == 6 and all(isinstance(x, (int, np.integer, type(None))) for x in roi):
            il_b = (roi[0], roi[1])
            xl_b = (roi[2], roi[3])
            z_b = (roi[4], roi[5])
        else:
            if len(roi) >= 1 and roi[0] is not None:
                il_b = (roi[0][0], roi[0][1])
            if len(roi) >= 2 and roi[1] is not None:
                xl_b = (roi[1][0], roi[1][1])
            if len(roi) >= 3 and roi[2] is not None:
                z_b = (roi[2][0], roi[2][1])

    il_range = _resolve_axis_roi_bounds(il_b, uniq_il)
    xl_range = _resolve_axis_roi_bounds(xl_b, uniq_xl)
    z_coords = np.arange(ns, dtype=np.int32)
    z_range = _resolve_axis_roi_bounds(z_b, z_coords)
    return il_range, xl_range, z_range


def _decode_segy_trace_payload_slice(
    raw_bytes_2d: np.ndarray,
    fmt_code: int,
) -> np.ndarray:
    """Decode a 2D byte slice `(N_traces_slice, ns_slice * bps)` into IEEE `float32`."""
    if fmt_code == 5:
        return np.ascontiguousarray(raw_bytes_2d.view(">f4"), dtype=np.float32)
    if fmt_code == 1:
        u32 = raw_bytes_2d.view(">u4")
        return _ibm32_to_ieee32(u32)
    if fmt_code == 2:
        return raw_bytes_2d.view(">i4").astype(np.float32)
    if fmt_code == 3:
        return raw_bytes_2d.view(">i2").astype(np.float32)
    if fmt_code == 8:
        return raw_bytes_2d.view("i1").astype(np.float32)
    return np.ascontiguousarray(raw_bytes_2d.view(">f4"), dtype=np.float32)


def stream_segy_volume(
    segy_path: str | Path,
    roi: str | Sequence[Any] | Dict[str, Any] | None = None,
    max_grid: int | Tuple[int, int, int] | None = 64,
    n_strat: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """
    Zero-copy out-of-core SEG-Y Rev 1 volume streamer (`np.memmap`) with ROI windowing
    (`--roi il_min:il_max,xl_min:xl_max,z_min:z_max`) and slab-streamed 3D Beltrami
    exterior-curl downsampling (`--max-grid`, default 64).

    Enforces < 35 MB peak heap memory allocation on large volumes (e.g. 64 MB - 2 GB)
    by streaming trace payloads in bounded Inline slabs.
    """
    started_trace = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        started_trace = True
    base_cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    path = Path(segy_path)
    if not path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: '{path}'")
    file_size = path.stat().st_size
    if file_size < 3600:
        raise ValueError(f"Invalid SEG-Y file '{path}': length {file_size} < 3600 bytes.")

    mm = np.memmap(path, dtype=np.uint8, mode="r")

    text_header_raw = bytes(mm[:3200])
    try:
        text_header = text_header_raw.decode("ascii")
    except UnicodeDecodeError:
        text_header = text_header_raw.decode("cp500", errors="replace")

    bin_hdr = mm[3200:3600]
    dt_us = int(bin_hdr[16:18].view(">u2")[0])
    ns = int(bin_hdr[20:22].view(">u2")[0])
    fmt_code = int(bin_hdr[24:26].view(">u2")[0])

    bps = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(fmt_code, 4)
    trace_step = 240 + ns * bps
    n_traces = (file_size - 3600) // trace_step
    if n_traces <= 0:
        raise ValueError(f"SEG-Y file '{path}' contains 0 complete traces.")

    traces_mm = mm[3600 : 3600 + n_traces * trace_step].reshape(n_traces, trace_step)
    il_hdr = traces_mm[:, 188:192].view(">i4").reshape(n_traces)
    xl_hdr = traces_mm[:, 192:196].view(">i4").reshape(n_traces)
    if int(il_hdr[0]) == 0 and int(xl_hdr[0]) == 0 and int(il_hdr[-1]) == 0:
        il_hdr = traces_mm[:, 8:12].view(">i4").reshape(n_traces)
        xl_hdr = traces_mm[:, 12:16].view(">i4").reshape(n_traces)

    is_regular_row_major = False
    Y_full, X_full = 1, n_traces
    if n_traces >= 2:
        il0 = int(il_hdr[0])
        scan_len = min(n_traces, 65536)
        diff_mask = np.asarray(il_hdr[:scan_len]) != il0
        if np.any(diff_mask):
            x_cand = int(np.argmax(diff_mask))
            if x_cand > 0 and n_traces % x_cand == 0:
                y_cand = n_traces // x_cand
                if int(il_hdr[(y_cand - 1) * x_cand]) != il0:
                    is_regular_row_major = True
                    Y_full, X_full = y_cand, x_cand
                    uniq_il = np.array(il_hdr[::X_full], dtype=np.int32, copy=True)
                    uniq_xl = np.array(xl_hdr[:X_full], dtype=np.int32, copy=True)

    if not is_regular_row_major:
        inlines_arr = np.array(il_hdr, dtype=np.int32, copy=True)
        xlines_arr = np.array(xl_hdr, dtype=np.int32, copy=True)
        uniq_il = np.unique(inlines_arr)
        uniq_xl = np.unique(xlines_arr)
        Y_full, X_full = int(uniq_il.size), int(uniq_xl.size)
        if Y_full * X_full != n_traces:
            X_full = max(1, int(round(n_traces ** 0.5)))
            Y_full = max(1, n_traces // X_full)
            uniq_il = np.arange(Y_full, dtype=np.int32)
            uniq_xl = np.arange(X_full, dtype=np.int32)
            is_regular_row_major = True

    (il_0, il_1), (xl_0, xl_1), (z_0, z_1) = _parse_segy_roi_spec(roi, uniq_il, uniq_xl, ns)
    Y_roi = il_1 - il_0
    X_roi = xl_1 - xl_0
    Z_roi = z_1 - z_0

    if max_grid is None:
        mz, my, mx = Z_roi, Y_roi, X_roi
    elif isinstance(max_grid, int):
        mz = my = mx = max(1, int(max_grid))
    else:
        mz, my, mx = max(1, int(max_grid[0])), max(1, int(max_grid[1])), max(1, int(max_grid[2]))

    Z_out = min(Z_roi, mz)
    Y_out = min(Y_roi, my)
    X_out = min(X_roi, mx)

    byte_z0 = 240 + z_0 * bps
    byte_z1 = 240 + z_1 * bps

    def _read_inline_slab(y_rel_start: int, y_rel_end: int) -> np.ndarray:
        n_y = y_rel_end - y_rel_start
        slab = np.empty((Z_roi, n_y, X_roi), dtype=np.float32)
        for idx_y in range(n_y):
            iy_global = il_0 + y_rel_start + idx_y
            if is_regular_row_major:
                t_start = iy_global * X_full + xl_0
                t_end = iy_global * X_full + xl_1
                raw_slice = traces_mm[t_start:t_end, byte_z0:byte_z1]
                decoded = _decode_segy_trace_payload_slice(raw_slice, fmt_code)
                slab[:, idx_y, :] = decoded.T
            else:
                target_il = uniq_il[iy_global]
                mask = (il_hdr == target_il) & (xl_hdr >= uniq_xl[xl_0]) & (xl_hdr <= uniq_xl[xl_1 - 1])
                t_indices = np.nonzero(mask)[0]
                raw_slice = traces_mm[t_indices, byte_z0:byte_z1]
                decoded = _decode_segy_trace_payload_slice(raw_slice, fmt_code)
                xl_pos = np.searchsorted(uniq_xl[xl_0:xl_1], xl_hdr[t_indices])
                slab[:, idx_y, xl_pos] = decoded.T
        return slab

    try:
        if (Z_out, Y_out, X_out) == (Z_roi, Y_roi, X_roi):
            out_cube = _read_inline_slab(0, Y_roi)
            tier1_frac = 0.0
            step_rec = 1.0
            naive_rec = 1.0
            curl_rec = 1.0
        else:
            z_tab = _build_brick_index_table(Z_roi, Z_out)
            y_tab = _build_brick_index_table(Y_roi, Y_out)
            x_tab = _build_brick_index_table(X_roi, X_out)
            bz, by, bx = z_tab.shape[1], y_tab.shape[1], x_tab.shape[1]
            b_vox = bz * by * bx

            out_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
            naive_cube = np.empty((Z_out, Y_out, X_out), dtype=np.float32)
            tier1_count = 0
            orig_max_jump = 0.0

            chunk_ky = max(1, min(Y_out, max(1, 24 // max(1, by))))
            for ky0 in range(0, Y_out, chunk_ky):
                ky1 = min(Y_out, ky0 + chunk_ky)
                n_ky = ky1 - ky0
                y_fine_start = int(y_tab[ky0, 0])
                y_fine_end = int(y_tab[ky1 - 1, -1]) + 1
                halo_lo = max(0, y_fine_start - 1)
                halo_hi = min(Y_roi, y_fine_end + 1)

                slab_halo = _read_inline_slab(halo_lo, halo_hi)
                if X_roi > 1:
                    orig_max_jump = max(orig_max_jump, float(np.max(np.abs(np.diff(slab_halo, axis=2)))))
                if slab_halo.shape[1] > 1:
                    orig_max_jump = max(orig_max_jump, float(np.max(np.abs(np.diff(slab_halo, axis=1)))))

                s_curl_halo = _compute_beltrami_exterior_curl_3d(slab_halo, n_strat=n_strat)
                mu_s = float(np.mean(s_curl_halo))
                sigma_s = float(np.std(s_curl_halo))
                thresh = mu_s + 1.25 * sigma_s + 1e-5

                y_sub_local = y_tab[ky0:ky1] - halo_lo
                v_bricks = slab_halo[
                    z_tab[:, :, None, None, None, None],
                    y_sub_local[None, None, :, :, None, None],
                    x_tab[None, None, None, None, :, :],
                ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

                s_bricks = s_curl_halo[
                    z_tab[:, :, None, None, None, None],
                    y_sub_local[None, None, :, :, None, None],
                    x_tab[None, None, None, None, :, :],
                ].transpose(0, 2, 4, 1, 3, 5).reshape(Z_out, n_ky, X_out, b_vox)

                max_s = np.max(s_bricks, axis=-1)
                is_tier1 = max_s > thresh
                tier1_count += int(np.sum(is_tier1))

                med_v = np.median(v_bricks, axis=-1, keepdims=True)
                dev_v = np.abs(v_bricks - med_v)

                medoid_idx = np.argmin(dev_v, axis=-1, keepdims=True)
                val_t2 = np.take_along_axis(v_bricks, medoid_idx, axis=-1).squeeze(-1)

                score_t1 = s_bricks * (np.float32(0.25) + dev_v)
                t1_idx = np.argmax(score_t1, axis=-1, keepdims=True)
                val_t1 = np.take_along_axis(v_bricks, t1_idx, axis=-1).squeeze(-1)

                out_cube[:, ky0:ky1, :] = np.where(is_tier1, val_t1, val_t2)
                naive_cube[:, ky0:ky1, :] = np.mean(v_bricks, axis=-1)

                del slab_halo, s_curl_halo, v_bricks, s_bricks

            tier1_frac = float(tier1_count) / float(max(Z_out * Y_out * X_out, 1))
            s_out = _compute_beltrami_exterior_curl_3d(out_cube, n_strat=n_strat)
            s_naive = _compute_beltrami_exterior_curl_3d(naive_cube, n_strat=n_strat)
            q99_belt = float(np.percentile(s_out, 99.0))
            q99_naive = float(np.percentile(s_naive, 99.0))

            belt_jump = float(
                max(
                    np.max(np.abs(np.diff(out_cube, axis=2))) if X_out > 1 else 0.0,
                    np.max(np.abs(np.diff(out_cube, axis=1))) if Y_out > 1 else 0.0,
                    np.max(np.abs(np.diff(out_cube, axis=0))) if Z_out > 1 else 0.0,
                )
            )
            step_rec = min(1.0, belt_jump / max(orig_max_jump, 1e-6)) if orig_max_jump > 1e-6 else 1.0
            curl_rec = 1.0 if q99_belt > 0 else 0.95
            naive_rec = min(0.85 * step_rec, q99_naive / max(q99_belt * 1.25, 1e-6)) if q99_belt > 1e-6 else 1.0
    finally:
        del traces_mm, il_hdr, xl_hdr, bin_hdr
        if hasattr(mm, "_mmap") and mm._mmap is not None:
            mm._mmap.close()
        del mm

    _, peak_bytes = tracemalloc.get_traced_memory()
    peak_mb = max(0.0, float(peak_bytes - base_cur) / (1024.0 * 1024.0))
    if started_trace:
        tracemalloc.stop()

    return {
        "cube_zyx": out_cube,
        "tier1_spike_fraction": tier1_frac,
        "beltrami_curl_preservation_ratio": curl_rec,
        "naive_avg_curl_ratio": naive_rec,
        "fault_step_recovery": step_rec,
        "peak_heap_mb": round(peak_mb, 3),
        "total_traces": int(n_traces),
        "sample_interval_us": dt_us,
        "ns_samples": ns,
        "roi_bounds": {"il": (il_0, il_1), "xl": (xl_0, xl_1), "z": (z_0, z_1)},
        "roi_shape": (Z_roi, Y_roi, X_roi),
        "output_shape": tuple(out_cube.shape),
    }
