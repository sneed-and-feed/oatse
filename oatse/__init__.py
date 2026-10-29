"""
OATSE: Ocean Acoustic Thermocline & Sonar Engine (v0.2.0, CC0-1.0 Public Domain).

A complete, standalone 3D/4D underwater acoustics, thermocline ray-refraction,
and seabed Synthetic Aperture Sonar (SAS) impedance inversion workstation powered
by the machine-checked mathematical physics primitives of OcclusionTriad.lean.
"""

from __future__ import annotations

__version__ = "0.2.0"
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
    tropical_kleene_closure,
    tropical_min_plus_matmul,
)
from oatse.studio import (
    export_portable_html_studio,
    export_portable_studio_html,
    generate_studio_html,
    render_sonar_showcase_figure,
    serialize_evaluation_payload,
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
    "tropical_kleene_closure",
    "tropical_min_plus_matmul",
    "export_portable_html_studio",
    "export_portable_studio_html",
    "generate_studio_html",
    "render_sonar_showcase_figure",
    "serialize_evaluation_payload",
]
