"""
oatse.acoustics: Ocean Acoustic Thermocline & Sonar Engine Acoustics Module.

Implements:
1. 9-term Mackenzie (1981) seawater sound-speed equation c(T, S, z) in m/s and km/s.
2. Canonical Munk (1974) SOFAR sound-speed velocity profile.
3. 3D Snell's-law critical grazing angle across Sonic Layer Depth (SLD) thermocline.
4. Multi-bounce surface-duct and bottom-bounce ray quantiles with 1/N_vox surgical weighting.
5. Acoustic shadow-zone exposure and transmission loss (oatse_shadow_tl_db in dB).
6. Coupled water-to-seabed acoustic impedance (oatse_zp = rho * V_p in km/s * g/cm^3).
7. Sub-bottom TTI elastic V_p(psi) using Thomsen anisotropy parameters (epsilon, delta, gamma).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard oceanographic & marine sediment acoustic catalog:
# (name, vp0_kms, vs0_kms, rho_gcc, epsilon, delta, gamma)
OCEANIC_ACOUSTIC_CATALOG: Tuple[Tuple[str, float, float, float, float, float, float], ...] = (
    ("surface_mixed_layer", 1.538, 0.000, 1.025, 0.00, 0.00, 0.00),
    ("cold_subthermocline_shadow", 1.462, 0.000, 1.028, 0.00, 0.00, 0.00),
    ("sofar_channel_axis", 1.480, 0.000, 1.030, 0.00, 0.00, 0.00),
    ("marine_turbidite_sand", 1.850, 0.980, 2.050, 0.09, 0.03, 0.11),
    ("shallow_gas_pocket", 0.720, 0.000, 1.380, 0.00, 0.00, 0.00),
    ("pelagic_clay", 1.650, 0.850, 1.600, 0.15, 0.05, 0.12),
    ("abyssal_basalt_bedrock", 5.200, 2.950, 2.850, 0.04, 0.02, 0.03),
    ("metallic_mine_cable_target", 5.950, 3.200, 4.025, 0.00, 0.00, 0.00),
)


@dataclass
class AcousticCubeBundle:
    """Coupled 3D ocean sound speed, transmission loss, and seabed acoustic impedance bundle."""

    sound_speed_kms: torch.Tensor
    shadow_tl_db: torch.Tensor
    acoustic_impedance_zp: torch.Tensor
    vp_tti_kms: torch.Tensor
    critical_angle_deg: torch.Tensor
    density_gcc: torch.Tensor

    @property
    def c(self) -> torch.Tensor:
        """Alias for sound speed volume in km/s."""
        return self.sound_speed_kms

    @property
    def oatse_shadow_tl_db(self) -> torch.Tensor:
        """Alias for shadow-zone transmission loss in dB."""
        return self.shadow_tl_db

    @property
    def oatse_zp(self) -> torch.Tensor:
        """Alias for acoustic impedance in km/s * g/cm^3."""
        return self.acoustic_impedance_zp

    @property
    def oatse_vp_tti(self) -> torch.Tensor:
        """Alias for TTI phase velocity in km/s."""
        return self.vp_tti_kms

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "sound_speed_kms": self.sound_speed_kms,
            "c": self.sound_speed_kms,
            "shadow_tl_db": self.shadow_tl_db,
            "oatse_shadow_tl_db": self.shadow_tl_db,
            "acoustic_impedance_zp": self.acoustic_impedance_zp,
            "oatse_zp": self.acoustic_impedance_zp,
            "vp_tti_kms": self.vp_tti_kms,
            "oatse_vp_tti": self.vp_tti_kms,
            "critical_angle_deg": self.critical_angle_deg,
            "density_gcc": self.density_gcc,
        }

    def __getitem__(self, key: str) -> Any:
        alias_map = {
            "c": "sound_speed_kms",
            "sound_speed": "sound_speed_kms",
            "tl": "shadow_tl_db",
            "tl_db": "shadow_tl_db",
            "oatse_shadow_tl_db": "shadow_tl_db",
            "zp": "acoustic_impedance_zp",
            "oatse_zp": "acoustic_impedance_zp",
            "vp_tti": "vp_tti_kms",
            "oatse_vp_tti": "vp_tti_kms",
        }
        attr = alias_map.get(key, key)
        return getattr(self, attr)


class MackenzieMunkAcousticEngine(nn.Module):
    """
    Ocean Acoustic Engine unifying Mackenzie (1981) seawater equations,
    Munk (1974) canonical SOFAR waveguides, Snell's-law critical refraction,
    and sub-bottom TTI elasticity.
    """

    def __init__(
        self,
        c0: float = 1.500,
        epsilon_munk: float = 0.00737,
        z_axis_m: float = 1300.0,
        b_scale_m: float = 1300.0,
        seawater_density_gcc: float = 1.025,
    ) -> None:
        super().__init__()
        self.c0 = float(c0)
        self.epsilon_munk = float(epsilon_munk)
        self.z_axis_m = float(z_axis_m)
        self.b_scale_m = float(b_scale_m)
        self.seawater_density_gcc = float(seawater_density_gcc)

    # ------------------------------------------------------------------------
    # 1. 9-Term Mackenzie (1981) Sound Speed Equation
    # ------------------------------------------------------------------------
    @staticmethod
    def mackenzie_sound_speed(
        temperature_c: Union[float, np.ndarray, torch.Tensor],
        salinity_ppt: Union[float, np.ndarray, torch.Tensor],
        depth_m: Union[float, np.ndarray, torch.Tensor],
        unit: str = "km/s",
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Evaluate the 9-term Mackenzie (1981) seawater sound-speed equation:
            c(T, S, D) = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
                         + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D^2
                         - 1.025e-2*T*(S - 35) - 7.139e-13*T*D^3
        where:
            T: Temperature in degrees Celsius (0 <= T <= 30 C)
            S: Salinity in parts per thousand / PSU (30 <= S <= 40 ppt)
            D: Depth in meters (0 <= D <= 8000 m)
        Returns:
            Sound speed c in 'm/s' or 'km/s' (default: 'km/s').
        """
        is_torch = isinstance(temperature_c, torch.Tensor) or isinstance(salinity_ppt, torch.Tensor) or isinstance(depth_m, torch.Tensor)
        is_numpy = isinstance(temperature_c, np.ndarray) or isinstance(salinity_ppt, np.ndarray) or isinstance(depth_m, np.ndarray)

        if is_torch:
            T = temperature_c if isinstance(temperature_c, torch.Tensor) else torch.as_tensor(temperature_c, dtype=torch.float32)
            S = salinity_ppt if isinstance(salinity_ppt, torch.Tensor) else torch.as_tensor(salinity_ppt, dtype=torch.float32)
            D = depth_m if isinstance(depth_m, torch.Tensor) else torch.as_tensor(depth_m, dtype=torch.float32)

            t1 = 1448.96
            t2 = 4.591 * T
            t3 = -5.304e-2 * (T ** 2)
            t4 = 2.374e-4 * (T ** 3)
            t5 = 1.340 * (S - 35.0)
            t6 = 1.630e-2 * D
            t7 = 1.675e-7 * (D ** 2)
            t8 = -1.025e-2 * T * (S - 35.0)
            t9 = -7.139e-13 * T * (D ** 3)

            c_ms = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
            return c_ms / 1000.0 if unit == "km/s" else c_ms

        elif is_numpy:
            T = np.asarray(temperature_c, dtype=np.float64)
            S = np.asarray(salinity_ppt, dtype=np.float64)
            D = np.asarray(depth_m, dtype=np.float64)

            t1 = 1448.96
            t2 = 4.591 * T
            t3 = -5.304e-2 * (T ** 2)
            t4 = 2.374e-4 * (T ** 3)
            t5 = 1.340 * (S - 35.0)
            t6 = 1.630e-2 * D
            t7 = 1.675e-7 * (D ** 2)
            t8 = -1.025e-2 * T * (S - 35.0)
            t9 = -7.139e-13 * T * (D ** 3)

            c_ms = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
            return c_ms / 1000.0 if unit == "km/s" else c_ms

        else:
            T = float(temperature_c)
            S = float(salinity_ppt)
            D = float(depth_m)

            c_ms = (
                1448.96
                + 4.591 * T
                - 5.304e-2 * (T ** 2)
                + 2.374e-4 * (T ** 3)
                + 1.340 * (S - 35.0)
                + 1.630e-2 * D
                + 1.675e-7 * (D ** 2)
                - 1.025e-2 * T * (S - 35.0)
                - 7.139e-13 * T * (D ** 3)
            )
            return float(c_ms / 1000.0 if unit == "km/s" else c_ms)

    @staticmethod
    def mackenzie_sound_speed_ms(T: Any, S: Any, D: Any) -> Any:
        return MackenzieMunkAcousticEngine.mackenzie_sound_speed(T, S, D, unit="m/s")

    @staticmethod
    def mackenzie_sound_speed_kms(T: Any, S: Any, D: Any) -> Any:
        return MackenzieMunkAcousticEngine.mackenzie_sound_speed(T, S, D, unit="km/s")

    # ------------------------------------------------------------------------
    # 2. Canonical Munk (1974) SOFAR Sound Speed Profile
    # ------------------------------------------------------------------------
    @staticmethod
    def munk_sound_speed(
        depth: Union[float, np.ndarray, torch.Tensor],
        c0: float = 1.500,
        epsilon_munk: float = 0.00737,
        z_axis: float = 1300.0,
        b_scale: float = 1300.0,
        unit: str = "km/s",
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Evaluate canonical Munk (1974) SOFAR profile:
            c(z) = c0 * [1 + epsilon * (eta - 1 + exp(-eta))]
            eta = 2 * (z - z_axis) / b_scale
        If depth is given in kilometers (< 50.0), z_axis and b_scale are automatically
        scaled if they exceed 50.0.
        """
        # Determine if depth is in meters or kilometers
        is_torch = isinstance(depth, torch.Tensor)
        is_numpy = isinstance(depth, np.ndarray)

        if is_torch:
            z = depth.float()
            # If z is in km (< 50) and z_axis is in m (> 50), adapt z_axis and b_scale to km
            z_max = float(z.max().item()) if z.numel() > 0 else 0.0
            za = z_axis / 1000.0 if (z_max < 50.0 and z_axis > 50.0) else z_axis
            bs = b_scale / 1000.0 if (z_max < 50.0 and b_scale > 50.0) else b_scale

            eta = 2.0 * (z - za) / max(bs, 1e-6)
            c = c0 * (1.0 + epsilon_munk * (eta - 1.0 + torch.exp(-eta)))
            if unit == "m/s" and c0 < 50.0:
                c = c * 1000.0
            elif unit == "km/s" and c0 > 50.0:
                c = c / 1000.0
            return c

        elif is_numpy:
            z = np.asarray(depth, dtype=np.float64)
            z_max = float(np.max(z)) if z.size > 0 else 0.0
            za = z_axis / 1000.0 if (z_max < 50.0 and z_axis > 50.0) else z_axis
            bs = b_scale / 1000.0 if (z_max < 50.0 and b_scale > 50.0) else b_scale

            eta = 2.0 * (z - za) / max(bs, 1e-6)
            c = c0 * (1.0 + epsilon_munk * (eta - 1.0 + np.exp(-eta)))
            if unit == "m/s" and c0 < 50.0:
                c = c * 1000.0
            elif unit == "km/s" and c0 > 50.0:
                c = c / 1000.0
            return c

        else:
            z = float(depth)
            za = z_axis / 1000.0 if (z < 50.0 and z_axis > 50.0) else z_axis
            bs = b_scale / 1000.0 if (z < 50.0 and b_scale > 50.0) else b_scale

            eta = 2.0 * (z - za) / max(bs, 1e-6)
            c = c0 * (1.0 + epsilon_munk * (eta - 1.0 + math.exp(-eta)))
            if unit == "m/s" and c0 < 50.0:
                c = c * 1000.0
            elif unit == "km/s" and c0 > 50.0:
                c = c / 1000.0
            return float(c)

    # ------------------------------------------------------------------------
    # 3. 3D Snell's-Law Critical Grazing Angle
    # ------------------------------------------------------------------------
    @staticmethod
    def critical_grazing_angle(
        c_volume: torch.Tensor,
        c_sld: Union[float, torch.Tensor] = 1.538,
        in_degrees: bool = True,
    ) -> torch.Tensor:
        """
        Evaluate 3D Snell's-law critical grazing angle across the Sonic Layer Depth:
            theta_c(z, y, x) = arccos( min(1.0, c(z, y, x) / c_SLD) )
        """
        ratio = torch.clamp(c_volume / float(c_sld), min=0.0, max=1.0)
        theta_rad = torch.acos(ratio)
        if in_degrees:
            return torch.rad2deg(theta_rad)
        return theta_rad

    # ------------------------------------------------------------------------
    # 4. Multi-Bounce Surface-Duct / Bottom-Bounce Ray Quantiles
    # ------------------------------------------------------------------------
    @staticmethod
    def compute_ray_quantiles(
        c_cube: torch.Tensor,
        sld_z: int = 6,
        seabed_z: int = 20,
        n_quantiles: int = 16,
    ) -> Dict[str, Any]:
        """
        Computes 3D surface-duct and bottom-bounce ray propagation quantiles
        with exact 1/N_vox surgical spatial weighting.
        """
        Z, Y, X = c_cube.shape
        N_vox = Z * Y * X
        device = c_cube.device

        # Surface duct mask (above Sonic Layer Depth)
        surface_duct = torch.zeros_like(c_cube, dtype=torch.bool)
        surface_duct[:sld_z, :, :] = True

        # Seabed sediment mask
        seabed_mask = torch.zeros_like(c_cube, dtype=torch.bool)
        seabed_mask[seabed_z:, :, :] = True

        # Acoustic shadow-zone mask (below SLD down to seabed or channel axis)
        shadow_mask = torch.zeros_like(c_cube, dtype=torch.bool)
        shadow_mask[sld_z:seabed_z, :, :] = True

        # Vertical sound speed gradient dc/dz
        dc_dz = torch.zeros_like(c_cube)
        if Z > 1:
            dc_dz[:-1, :, :] = c_cube[1:, :, :] - c_cube[:-1, :, :]
            dc_dz[-1, :, :] = dc_dz[-2, :, :]

        # Normalized ray slowness quantiles
        slowness = 1.0 / torch.clamp(c_cube, min=0.1)
        flat_slow = slowness.view(-1)
        q_probs = torch.linspace(0.05, 0.95, n_quantiles, device=device)
        quantiles = torch.quantile(flat_slow, q_probs)

        # 1/N_vox surgical spatial weighting
        surgical_weights = torch.full((Z, Y, X), 1.0 / float(N_vox), device=device, dtype=torch.float32)

        return {
            "surface_duct_mask": surface_duct,
            "seabed_mask": seabed_mask,
            "shadow_mask": shadow_mask,
            "dc_dz": dc_dz,
            "slowness_quantiles": quantiles,
            "surgical_ray_weights": surgical_weights,
            "n_quantiles": n_quantiles,
            "n_voxels": N_vox,
        }

    # ------------------------------------------------------------------------
    # 5. Acoustic Shadow-Zone Exposure & Transmission Loss (dB)
    # ------------------------------------------------------------------------
    @staticmethod
    def compute_shadow_transmission_loss(
        c_cube: torch.Tensor,
        source_zyx: Tuple[int, int, int] = (2, 12, 12),
        sld_z: int = 6,
        seabed_z: int = 20,
        freq_khz: float = 3.5,
        absorption_db_km: float = 0.25,
    ) -> torch.Tensor:
        """
        Evaluate 3D acoustic transmission loss (TL in dB) including:
        - Geometric spreading (cylindrical in surface duct, spherical elsewhere)
        - Seawater volume absorption
        - Acoustic shadow-zone penetration penalty beneath Sonic Layer Depth (SLD).
        """
        Z, Y, X = c_cube.shape
        device = c_cube.device
        sz = min(Z - 1, max(0, int(source_zyx[0])))
        sy = min(Y - 1, max(0, int(source_zyx[1])))
        sx = min(X - 1, max(0, int(source_zyx[2])))

        # 3D spatial grid coordinates
        zz, yy, xx = torch.meshgrid(
            torch.arange(Z, device=device, dtype=torch.float32),
            torch.arange(Y, device=device, dtype=torch.float32),
            torch.arange(X, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Radial distance in grid units (assume 1 grid unit ~ 50 m or normalized range)
        dx_km = 0.050  # 50 m per voxel
        dz_km = 0.025  # 25 m depth per voxel
        r_km = torch.sqrt(
            ((zz - float(sz)) * dz_km) ** 2
            + ((yy - float(sy)) * dx_km) ** 2
            + ((xx - float(sx)) * dx_km) ** 2
        )
        r_safe = torch.clamp(r_km, min=0.01)

        # Base geometric spreading
        # Inside surface duct (z <= sld_z), cylindrical spreading: TL = 10 log10(R) + const
        # Outside surface duct, spherical spreading: TL = 20 log10(R) + const
        tl_cyl = 10.0 * torch.log10(r_safe / 0.01) + 40.0
        tl_sph = 20.0 * torch.log10(r_safe / 0.01) + 40.0

        is_duct = zz <= float(sld_z)
        tl_base = torch.where(is_duct, tl_cyl, tl_sph) + absorption_db_km * r_km

        # Shadow zone excess attenuation beneath SLD
        # Cold sub-thermocline shadow zone refracts rays downward away from target
        c_sld = float(c_cube[sld_z, sy, sx].item()) if sld_z < Z else 1.538
        depth_below_sld = torch.clamp(zz - float(sld_z), min=0.0)
        c_ratio = torch.clamp(c_cube / max(c_sld, 0.1), min=0.5, max=1.2)
        refraction_penalty = 12.0 * (1.0 - torch.clamp(c_ratio, max=1.0))

        shadow_excess = 3.5 * torch.pow(depth_below_sld, 1.25) + refraction_penalty
        # Sub-bottom sediment penetration loss
        seabed_excess = 25.0 * torch.clamp(zz - float(seabed_z), min=0.0)

        tl_total = tl_base + shadow_excess + seabed_excess
        return torch.clamp(tl_total, min=30.0, max=140.0)

    # ------------------------------------------------------------------------
    # 6. Coupled Water-to-Seabed Acoustic Impedance (Z_p)
    # ------------------------------------------------------------------------
    @staticmethod
    def compute_acoustic_impedance(
        vp_kms: torch.Tensor,
        rho_gcc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Coupled acoustic P-wave impedance:
            Z_p = rho * V_p  (in km/s * g/cm^3)
        """
        return rho_gcc * vp_kms

    @staticmethod
    def compute_seabed_reflection_coefficient(
        zp_water: Union[float, torch.Tensor],
        zp_seabed: Union[float, torch.Tensor],
    ) -> Union[float, torch.Tensor]:
        """
        Normal-incidence acoustic pressure reflection coefficient:
            R_p = (Z_seabed - Z_water) / (Z_seabed + Z_water)
        """
        return (zp_seabed - zp_water) / (zp_seabed + zp_water + 1e-12)

    # ------------------------------------------------------------------------
    # 7. Sub-Bottom TTI Elastic Phase Velocities V_p(psi)
    # ------------------------------------------------------------------------
    @staticmethod
    def compute_vp_tti(
        vp0_kms: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor,
        dip_deg: Union[float, torch.Tensor] = 0.0,
        azimuth_deg: Union[float, torch.Tensor] = 0.0,
        ray_direction_zyx: Tuple[float, float, float] = (0.5, 0.0, 0.8660254),
    ) -> torch.Tensor:
        """
        Evaluate Thomsen (1986) weak-anisotropy TTI phase velocity V_p(psi):
            V_p(psi) = V_p0 * (1 + delta * sin^2(psi) * cos^2(psi) + epsilon * sin^4(psi))
        where psi is the angle between propagation direction and tilted symmetry axis normal.
        """
        device = vp0_kms.device
        orig_dtype = vp0_kms.dtype

        # Bedding dip and azimuth
        theta = torch.deg2rad(torch.as_tensor(dip_deg, device=device, dtype=torch.float64))
        phi = torch.deg2rad(torch.as_tensor(azimuth_deg, device=device, dtype=torch.float64))

        # Symmetry axis unit normal vector b = (bz, by, bx)
        bz = torch.cos(theta)
        by = torch.sin(theta) * torch.sin(phi)
        bx = torch.sin(theta) * torch.cos(phi)

        # Normalized ray direction
        rz, ry, rx = ray_direction_zyx
        rn = math.sqrt(rz * rz + ry * ry + rx * rx)
        rz, ry, rx = rz / rn, ry / rn, rx / rn

        # cos(psi) = b . ray
        cos_psi = torch.clamp(bz * rz + by * ry + bx * rx, min=-1.0, max=1.0)
        cos2_psi = cos_psi * cos_psi
        sin2_psi = torch.clamp(1.0 - cos2_psi, min=0.0, max=1.0)
        sin4_psi = sin2_psi * sin2_psi

        vp0 = vp0_kms.to(torch.float64)
        eps = epsilon.to(torch.float64)
        dlt = delta.to(torch.float64)

        vp_tti = vp0 * (1.0 + dlt * sin2_psi * cos2_psi + eps * sin4_psi)
        return vp_tti.to(orig_dtype)

    @staticmethod
    def compute_tti_bundle(
        vp0_kms: torch.Tensor,
        vs0_kms: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        dip_deg: Union[float, torch.Tensor] = 0.0,
        azimuth_deg: Union[float, torch.Tensor] = 0.0,
        ray_direction_zyx: Tuple[float, float, float] = (0.5, 0.0, 0.8660254),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate full Thomsen TTI phase velocities (V_p, V_sv, V_sh).
        """
        device = vp0_kms.device
        orig_dtype = vp0_kms.dtype

        theta = torch.deg2rad(torch.as_tensor(dip_deg, device=device, dtype=torch.float64))
        phi = torch.deg2rad(torch.as_tensor(azimuth_deg, device=device, dtype=torch.float64))

        bz = torch.cos(theta)
        by = torch.sin(theta) * torch.sin(phi)
        bx = torch.sin(theta) * torch.cos(phi)

        rz, ry, rx = ray_direction_zyx
        rn = math.sqrt(rz * rz + ry * ry + rx * rx)
        rz, ry, rx = rz / rn, ry / rn, rx / rn

        cos_psi = torch.clamp(bz * rz + by * ry + bx * rx, min=-1.0, max=1.0)
        cos2_psi = cos_psi * cos_psi
        sin2_psi = torch.clamp(1.0 - cos2_psi, min=0.0, max=1.0)
        sin4_psi = sin2_psi * sin2_psi

        vp0 = vp0_kms.to(torch.float64)
        vs0 = vs0_kms.to(torch.float64)
        eps = epsilon.to(torch.float64)
        dlt = delta.to(torch.float64)
        gam = gamma.to(torch.float64)

        vp_ratio_sq = torch.clamp((vp0 / torch.clamp(vs0, min=1e-3)) ** 2, min=1.0, max=25.0)

        vp_tti = vp0 * (1.0 + dlt * sin2_psi * cos2_psi + eps * sin4_psi)
        vsv_tti = vs0 * (1.0 + vp_ratio_sq * (eps - dlt) * sin2_psi * cos2_psi)
        vsh_tti = vs0 * (1.0 + gam * sin2_psi)

        return vp_tti.to(orig_dtype), vsv_tti.to(orig_dtype), vsh_tti.to(orig_dtype)

    # ------------------------------------------------------------------------
    # 8. Full Forward Bundle Builder for 3D Volume
    # ------------------------------------------------------------------------
    def forward(
        self,
        sound_speed_3d: torch.Tensor,
        density_3d: Optional[torch.Tensor] = None,
        epsilon_3d: Optional[torch.Tensor] = None,
        delta_3d: Optional[torch.Tensor] = None,
        sld_z: int = 6,
        seabed_z: int = 20,
        source_zyx: Tuple[int, int, int] = (2, 12, 12),
        dip_deg: float = 0.0,
        azimuth_deg: float = 0.0,
    ) -> AcousticCubeBundle:
        """
        Evaluate full coupled ocean acoustics bundle on a 3D grid `(Z, Y, X)`.
        """
        c = sound_speed_3d.float()
        device = c.device

        if density_3d is None:
            # Default ocean seawater + sediment density profile
            Z, Y, X = c.shape
            rho = torch.full_like(c, self.seawater_density_gcc)
            if seabed_z < Z:
                rho[seabed_z:, :, :] = 2.05  # marine sediment density
        else:
            rho = density_3d.float()

        if epsilon_3d is None:
            eps = torch.zeros_like(c)
        else:
            eps = epsilon_3d.float()

        if delta_3d is None:
            dlt = torch.zeros_like(c)
        else:
            dlt = delta_3d.float()

        # 1. Critical grazing angle
        c_sld = float(c[sld_z, source_zyx[1], source_zyx[2]].item()) if sld_z < c.shape[0] else 1.538
        crit_angle = self.critical_grazing_angle(c, c_sld=c_sld, in_degrees=True)

        # 2. Transmission loss in dB
        tl_db = self.compute_shadow_transmission_loss(
            c, source_zyx=source_zyx, sld_z=sld_z, seabed_z=seabed_z
        )

        # 3. Acoustic impedance Z_p
        zp = self.compute_acoustic_impedance(c, rho)

        # 4. Sub-bottom TTI phase velocity
        vp_tti = self.compute_vp_tti(c, eps, dlt, dip_deg=dip_deg, azimuth_deg=azimuth_deg)

        return AcousticCubeBundle(
            sound_speed_kms=c,
            shadow_tl_db=tl_db,
            acoustic_impedance_zp=zp,
            vp_tti_kms=vp_tti,
            critical_angle_deg=crit_angle,
            density_gcc=rho,
        )
