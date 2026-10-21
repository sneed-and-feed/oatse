"""
OATSE: Ocean Acoustic Thermocline & Sonar Engine (v0.1.0, CC0-1.0 Public Domain).

A complete, standalone 3D/4D underwater acoustics, thermocline ray-refraction,
and seabed Synthetic Aperture Sonar (SAS) impedance inversion workstation powered
by the machine-checked mathematical physics primitives of OcclusionTriad.lean.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "OATSE Development Team"
__license__ = "CC0-1.0 Universal Public Domain Dedication"

from oatse.acoustics import (
    OCEANIC_ACOUSTIC_CATALOG,
    AcousticCubeBundle,
    MackenzieMunkAcousticEngine,
)
from oatse.engine import (
    BeltramiCastShell,
    BeltramiCastShellResult,
    ContinuousSpectralOceanField3D,
    OATSEModel,
    RoPE3DOceanMedoidPooler,
    SkewSymmetricThermoclineTriad,
    SlowFastOceanCascade,
    TemporalSonarContinuitySplitter,
    TheoremAuditResult,
    build_kelvin_mandel_so6_rotor,
    build_so3_from_dip_azimuth,
    build_vti_kelvin_stiffness_tensor,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "OCEANIC_ACOUSTIC_CATALOG",
    "AcousticCubeBundle",
    "MackenzieMunkAcousticEngine",
    "BeltramiCastShell",
    "BeltramiCastShellResult",
    "ContinuousSpectralOceanField3D",
    "OATSEModel",
    "RoPE3DOceanMedoidPooler",
    "SkewSymmetricThermoclineTriad",
    "SlowFastOceanCascade",
    "TemporalSonarContinuitySplitter",
    "TheoremAuditResult",
    "build_kelvin_mandel_so6_rotor",
    "build_so3_from_dip_azimuth",
    "build_vti_kelvin_stiffness_tensor",
]
