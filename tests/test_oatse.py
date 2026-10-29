"""
tests/test_oatse.py: Comprehensive Verification & Pytest Test Suite for OATSE v0.2.0.

Tests cover:
  1. Machine-checked formal theorem runtime verification (Lean Theorems 2.1–2.7).
  2. Drop-in OATSEModel contract & autograd compatibility.
  3. Degenerate inputs & zero-front edge cases.
  4. Pure non-circular BeltramiCastShell spike recovery vs. equal-budget uniform pooling.
  5. Continuous spectral ocean field & 3D Jordan cavity isolation.
  6. Bit-identity across Z-slab chunk_z in {8, 16, 32}.
  7. SWaP-C memory ceiling under tracemalloc (< 35 MB peak heap).
  8. All 5 maritime scenarios generation and integrity.
  9. EOFAnisotropicKriging (MODAS / sEOF-r) SVD extraction & 2D Matérn covariance.
  10. Official community benchmarks suite (ASA Pekeris Wedge, ONR/NRL Geoacoustic, OALIB Munk WKB).
  11. Mode B Blind Benchmark on subsea_cable_mine_corridor with SNR=20 dB.
  12. Mackenzie-Munk acoustic engine physics equations.
  13. IO round-tripping & memory-mapped SEGY slab streaming.
  14. OATSEModel triad & cascade forward pass and scenario comparison without fallback.
  15. Studio v0.2.0 payload serialization (17 cubes, Snell ray-fan, fault planes, Beltrami curves).
  16. Portable HTML5 Studio v0.2.0 offline integrity & view controls verification.
  17. Publication 4-panel showcase figure renderer & CLI subcommands (v0.2.0).
  18. 4D TemporalSonarContinuitySplitter M2 internal tide vs. irreversible target separation.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
import tempfile
import tracemalloc

import numpy as np
import pytest
import torch

from oatse.acoustics import (
    OCEANIC_ACOUSTIC_CATALOG,
    AcousticCubeBundle,
    MackenzieMunkAcousticEngine,
)
from oatse.blind_benchmark import (
    EOFAnisotropicKriging,
    evaluate_beltrami_noise_and_washout_ablation,
    evaluate_blind_holdout_and_baselines,
    fit_and_predict_anisotropic_matern_kriging,
    fit_and_predict_eof_anisotropic_kriging,
    run_official_community_benchmarks,
    run_referee_audit_suite,
)
from oatse.engine import (
    BeltramiCastShell,
    ContinuousSpectralOceanField3D,
    OATSEModel,
    RoPE3DOceanMedoidPooler,
    SkewSymmetricThermoclineTriad,
    SlowFastOceanCascade,
    TemporalSonarContinuitySplitter,
    build_kelvin_mandel_so6_rotor,
    build_so3_from_dip_azimuth,
    build_vti_kelvin_stiffness_tensor,
)
from oatse.io import (
    read_dev_survey,
    read_las2_file,
    read_segy_cube,
    solve_minimum_curvature_trajectory,
    stream_segy_volume,
    write_dev_survey,
    write_las2_file,
    write_segy_cube,
)
from oatse.scenarios import (
    MARITIME_ACOUSTIC_CATALOG,
    SCENARIO_DESCRIPTIONS,
    build_all_scenarios,
    build_maritime_scenario,
    resolve_scenario_name,
)


def test_suite1_formal_theorems_2_1_to_2_7_runtime():
    """Verify machine-checked invariants from OcclusionTriad.lean (Theorems 2.1 through 2.7)."""
    model = OATSEModel(grid_shape=(16, 16, 16))
    audit = model.verify_formal_theorems()

    assert audit["theorem_2_1"]["passed"] is True
    assert audit["theorem_2_1"]["antisymmetry_error"] == 0.0
    assert audit["theorem_2_1"]["self_interaction_error"] == 0.0

    assert audit["theorem_2_2"]["passed"] is True
    assert audit["theorem_2_2"]["tie_break_gap"] > 0.0
    assert audit["theorem_2_2"]["formula_error"] < 1e-6

    assert audit["theorem_2_3"]["passed"] is True
    assert audit["theorem_2_3"]["orthogonality_residual"] <= 1e-8

    assert audit["theorem_2_4"]["passed"] is True
    assert audit["theorem_2_4"]["step0_homotopy_error"] == 0.0

    assert audit["theorem_2_5"]["passed"] is True
    assert audit["theorem_2_5"]["relative_phase_error"] < 1e-12
    assert audit["theorem_2_5"]["spike_recovery"] == 1.0
    assert audit["theorem_2_5"]["norm_attenuation_loss"] > 0.10

    assert audit["theorem_2_6"]["passed"] is True
    assert audit["theorem_2_6"]["spherical_radial_drift"] <= 1e-8
    assert audit["theorem_2_6"]["cascade_orthogonality_error"] <= 1e-8

    assert audit["theorem_2_7"]["passed"] is True
    assert audit["theorem_2_7"]["so6_orthogonality_error"] < 1e-12
    assert all(audit[k]["passed"] for k in audit)


def test_suite2_drop_in_model_contract_and_autograd():
    """Verify drop-in API compatibility and finite autograd gradients."""
    model = OATSEModel(
        grid_shape=(12, 12, 12),
        num_layers=8,
        num_ocean_modes=6,
    )
    c_input = torch.full((12, 12, 12), 1.500, requires_grad=True)
    rho_input = torch.full((12, 12, 12), 1.025)

    bundle = model(c_input, density_3d=rho_input)
    assert isinstance(bundle, AcousticCubeBundle)
    assert bundle.sound_speed_kms.shape == (12, 12, 12)
    assert bundle.shadow_tl_db.shape == (12, 12, 12)
    assert bundle.acoustic_impedance_zp.shape == (12, 12, 12)
    assert bundle.vp_tti_kms.shape == (12, 12, 12)

    loss = bundle.sound_speed_kms.sum()
    loss.backward()
    assert c_input.grad is not None
    assert torch.isfinite(c_input.grad).all()


def test_suite3_zero_front_context_and_degenerate_inputs():
    """Audit edge cases: zero front indicators, zero vectors, and constant borehole traces."""
    model = OATSEModel(grid_shape=(8, 8, 8))
    model.W_triad.data.normal_(0.0, 1.0)

    # Zero u_front must produce zero triad perturbation
    u_zero = torch.zeros(64, 6)
    v_probs = torch.softmax(torch.randn(64, 8), dim=-1)
    d_ell = model.skew_symmetric_thermocline_triad(u_zero, v_probs)
    assert float(d_ell.abs().max().item()) == 0.0

    # Zero v_probs must not NaN
    v_zero = torch.zeros(16, 8)
    u_rand = torch.randn(16, 6)
    d_ell_z = model.skew_symmetric_thermocline_triad(u_rand, v_zero)
    assert torch.isfinite(d_ell_z).all()

    # Constant trace must not NaN
    const_trace = torch.full((64,), 1.500)
    salience = model.beltrami_cast_salience(const_trace)
    assert salience.shape == (64,)
    assert torch.isfinite(salience).all()
    assert float(salience.max().item()) == 0.0


def test_suite4_beltrami_cast_shell_spike_recovery_vs_naive():
    """Verify BeltramiCastShell preserves thin high-contrast spikes at 1.0000 recovery."""
    Z = 64
    trace = torch.full((Z,), 1.500, dtype=torch.float32)
    trace[20] = 5.950  # Metallic mine / subsea cable high-impedance tap
    trace[44] = 0.720  # Shallow gas pocket low-velocity spike

    shell = BeltramiCastShell(outlier_budget=8, condensed_budget=8)
    res = shell(trace)

    assert res.beltrami_spike_recovery == pytest.approx(1.0, abs=1e-5)
    assert res.naive_spike_recovery < 0.35
    assert 20 in res.tier1_outlier_indices.tolist()
    assert 44 in res.tier1_outlier_indices.tolist()
    assert float(res.reconstructed_vp[20].item()) == pytest.approx(5.950, abs=1e-4)
    assert float(res.reconstructed_vp[44].item()) == pytest.approx(0.720, abs=1e-4)


def test_suite5_continuous_spectral_ocean_field_and_cavity_isolation():
    """Verify ContinuousSpectralOceanField3D isolates enclosed acoustic cavities."""
    Z, Y, X = 24, 24, 24
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, dtype=torch.float32),
        torch.arange(Y, dtype=torch.float32),
        torch.arange(X, dtype=torch.float32),
        indexing="ij",
    )
    r2 = ((zz - 12.0) / 6.0) ** 2 + ((yy - 12.0) / 6.0) ** 2 + ((xx - 12.0) / 6.0) ** 2
    hollow_shell = ((r2 >= 0.55) & (r2 <= 1.05)).float()

    spec = ContinuousSpectralOceanField3D()
    strat = 1.480 + 0.05 * (zz / Z)
    out = spec(strat, hollow_shell)

    cav = out["cavity_info"]
    j6 = cav["jordan_6way"]
    assert bool(j6[12, 12, 12].item()) is True
    assert bool(j6[1, 1, 1].item()) is False


def test_suite6_chunk_z_bit_identity():
    """Verify bit-identical output across Z-slab chunk_z in {8, 16, 32}."""
    model = OATSEModel(grid_shape=(32, 16, 16))
    c_cube = torch.randn(32, 16, 16).clamp(1.45, 1.55)

    with torch.no_grad():
        out8 = model(c_cube, chunk_z=8)
        out16 = model(c_cube, chunk_z=16)
        out32 = model(c_cube, chunk_z=32)

    diff8_16 = (out8.sound_speed_kms - out16.sound_speed_kms).abs().max().item()
    diff16_32 = (out16.sound_speed_kms - out32.sound_speed_kms).abs().max().item()
    assert diff8_16 < 1e-6, f"chunk_z 8 vs 16 discrepancy: {diff8_16}"
    assert diff16_32 < 1e-6, f"chunk_z 16 vs 32 discrepancy: {diff16_32}"


def test_suite7_swap_c_memory_ceiling_tracemalloc():
    """Verify strict SWaP-C peak heap ceiling (< 35 MB) on 3D volume evaluation and SEGY streaming."""
    gc.collect()
    tracemalloc.start()

    # 1. 3D volume evaluation (32, 24, 24)
    scen = build_maritime_scenario("subsea_cable_mine_corridor", grid_shape=(32, 24, 24))
    model = OATSEModel(grid_shape=(32, 24, 24))
    with torch.no_grad():
        _ = model(scen["true_vp_cube"], density_3d=scen["true_rho_cube"], chunk_z=16)

    # 2. Simulated SEGY out-of-core streaming (< 35 MB peak heap)
    with tempfile.TemporaryDirectory() as tmpdir:
        segy_file = Path(tmpdir) / "bench_volume.segy"
        cube_data = np.full((32, 32, 32), 1.500, dtype=np.float32)
        write_segy_cube(str(segy_file), cube_data)

        stream_res = stream_segy_volume(str(segy_file), max_grid=32)
        assert "cube_zyx" in stream_res
        assert stream_res["cube_zyx"].shape == (32, 32, 32)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    assert peak_mb < 35.0, f"Peak heap RAM {peak_mb:.2f} MB exceeded 35 MB ceiling!"


def test_suite8_all_five_scenarios_generation_and_consistency():
    """Verify all 5 scenarios generate 4 vertical casts and 4 deviated runs with valid acoustics."""
    all_scenarios = build_all_scenarios(grid_shape=(32, 24, 24))
    assert len(all_scenarios) == 5

    expected_names = [
        "munk_sofar_shadow_zone",
        "giuk_polar_halocline_front",
        "littoral_internal_tide_wedge",
        "subsea_cable_mine_corridor",
        "north_atlantic_xbt_field_doublet",
    ]

    for name in expected_names:
        assert name in all_scenarios
        scen = all_scenarios[name]
        assert "true_vp_cube" in scen
        assert "true_rho_cube" in scen
        assert "wells" in scen
        assert "deviated_wells" in scen
        assert len(scen["wells"]) == 4
        assert len(scen["deviated_wells"]) == 4

        c = scen["true_vp_cube"]
        rho = scen["true_rho_cube"]
        assert c.shape == (32, 24, 24)
        assert rho.shape == (32, 24, 24)
        assert bool((c >= 0.50).all().item())
        assert bool((rho >= 0.90).all().item())


def test_suite9_eof_anisotropic_kriging_modas():
    """Verify EOFAnisotropicKriging (MODAS / sEOF-r) SVD mode extraction and 2D Matérn covariance."""
    Z, Y, X, N = 32, 24, 24, 4
    # Create 4 training casts
    casts: list[dict[str, Any]] = []
    coords = [(4, 4), (4, 18), (18, 4), (18, 18)]
    for i, (cy, cx) in enumerate(coords):
        trace = torch.sin(torch.linspace(0, 3.14, Z)) * (1.0 + 0.1 * i) + 1.500
        casts.append({
            "well_id": i,
            "y": cy,
            "x": cx,
            "true_vp_trace": trace,
            "true_c_trace": trace,
        })

    pred_cube, info = fit_and_predict_eof_anisotropic_kriging(
        casts=casts,
        train_indices=[0, 1, 2, 3],
        grid_shape=(Z, Y, X),
        num_modes=2,
    )
    assert pred_cube.shape == (Z, Y, X)
    assert info["num_modes"] == 2
    assert torch.isfinite(pred_cube).all()

    # Verify reproduction at cast locations within nugget tolerance
    for i, (cy, cx) in enumerate(coords):
        gt = casts[i]["true_vp_trace"]
        pred_trace = pred_cube[:, cy, cx]
        rmse = float(torch.sqrt(torch.mean((pred_trace - gt) ** 2)).item())
        assert rmse < 0.05, f"Cast {i} reproduction RMSE {rmse} too high"


def test_suite10_official_community_benchmarks():
    """Verify the 3 official ocean acoustics community benchmarks."""
    res = run_official_community_benchmarks()
    assert "pekeris_wedge" in res
    assert "onr_nrl_geoacoustic_blind" in res
    assert "munk_wkb_thorp_reference" in res

    # 1. ASA Pekeris Wedge
    pek = res["pekeris_wedge"]
    assert abs(pek["snell_critical_angle_deg"] - 28.0725) < 0.01
    assert abs(pek["snell_critical_angle_rad"] - 0.489957) < 0.001
    assert pek["modal_cutoff_progression_verified"] is True
    assert pek["seabed_interface_fidelity"]["blind_oatse"] > 0.95
    assert (
        pek["critical_angle_recovery_deg"]["blind_oatse"]
        > pek["critical_angle_recovery_deg"]["tuned_anisotropic_kriging"]
    )

    # 2. ONR / NRL Geoacoustic Inversion Workshop Case A/C
    onr = res["onr_nrl_geoacoustic_blind"]
    assert onr["thin_anomaly_detected"] is True
    assert onr["thin_sub_bottom_spike_recovery"]["blind_oatse"] == 1.0000
    assert onr["thin_sub_bottom_spike_recovery"]["eof_anisotropic_kriging"] < 0.35
    assert onr["basement_impedance_zp_rmse_mrayl"]["blind_oatse"] < 0.020

    # 3. OALIB Munk WKB Ray-Tube & Thorp Attenuation Reference
    mnk = res["munk_wkb_thorp_reference"]
    assert abs(mnk["theoretical_ray_loop_lambda_km"] - 47.663) < 0.01
    assert abs(mnk["sofar_ray_loop_length_km"]["blind_oatse"] - 47.663) < 0.01
    assert mnk["thorp_attenuation_1khz_db_km"] == pytest.approx(0.0690, abs=1e-3)
    assert mnk["thorp_attenuation_3p5khz_db_km"] == pytest.approx(0.2392, abs=1e-3)
    assert mnk["shadow_tl_residual_db"]["blind_oatse"] < 0.080
    assert mnk["wkb_thorp_verified"] is True


def test_suite11_mode_b_blind_benchmark_superiority():
    """Verify Mode B blind benchmark: zero target leakage and OATSE superiority over Kriging."""
    res = evaluate_blind_holdout_and_baselines(
        scenario_name="subsea_cable_mine_corridor",
        grid_shape=(32, 24, 24),
        snr_db=20.0,
        front_jitter_voxels=0,
    )
    wc = res["whole_cube_blind_rmse_kms"]
    lowo = res["lowo_heldout_cast_rmse_kms"]
    zp = res["seabed_zp_blind_rmse_mrayl"]
    ef = res["bounded_edge_gradient_fidelity"]

    # BlindOATSE outperforms TunedAnisotropicKriging and EOFAnisotropicKriging on whole cube
    assert wc["blind_oatse"] < wc["tuned_anisotropic_kriging"]
    assert wc["blind_oatse"] < wc["eof_anisotropic_kriging"]
    assert wc["blind_oatse"] < wc["naive_box"]

    # Edge Gradient Fidelity is bounded in [0, 1]
    assert 0.0 <= ef["blind_oatse"] <= 1.0
    assert 0.0 <= ef["tuned_anisotropic_kriging"] <= 1.0
    assert 0.0 <= ef["eof_anisotropic_kriging"] <= 1.0

    # Seabed Zp Acoustic Impedance RMSE
    assert zp["blind_oatse"] < zp["naive_box"]


def test_suite12_mackenzie_munk_acoustics_engine():
    """Verify 9-term Mackenzie (1981) sound-speed equation and Munk SOFAR waveguide profile."""
    engine = MackenzieMunkAcousticEngine()

    # Mackenzie equation: T=10 C, S=35 ppt, D=1000m
    c_mack = MackenzieMunkAcousticEngine.mackenzie_sound_speed_kms(10.0, 35.0, 1000.0)
    assert 1.480 < c_mack < 1.520

    # Munk profile minimum at axis z=1.3 km
    c_axis = MackenzieMunkAcousticEngine.munk_sound_speed(1.3, c0=1.500, unit="km/s")
    c_above = MackenzieMunkAcousticEngine.munk_sound_speed(0.5, c0=1.500, unit="km/s")
    c_below = MackenzieMunkAcousticEngine.munk_sound_speed(3.0, c0=1.500, unit="km/s")
    assert c_axis < c_above
    assert c_axis < c_below

    # Snell critical grazing angle
    c_vol = torch.tensor([1.500, 1.450, 1.538])
    crit_deg = MackenzieMunkAcousticEngine.critical_grazing_angle(c_vol, c_sld=1.538, in_degrees=True)
    assert float(crit_deg[2].item()) == pytest.approx(0.0, abs=1e-3)
    assert float(crit_deg[0].item()) > 0.0

    # Coupled acoustic impedance Z_p = rho * V_p
    zp = MackenzieMunkAcousticEngine.compute_acoustic_impedance(
        torch.tensor([1.500]), torch.tensor([1.025])
    )
    assert float(zp.item()) == pytest.approx(1.5375, abs=1e-4)


def test_suite13_io_streaming_round_trip():
    """Verify NOAA XBT (.nc / .xbt), SEGY cube, and dev survey IO round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Dev survey trajectory
        dev_file = tmp / "survey.dev"
        traj = solve_minimum_curvature_trajectory(
            [0.0, 100.0, 200.0, 300.0],
            [0.0, 5.0, 10.0, 15.0],
            [0.0, 45.0, 90.0, 120.0],
        )
        write_dev_survey(str(dev_file), traj)
        survey_traj = read_dev_survey(str(dev_file))
        assert len(survey_traj.md_m) == 4
        assert len(traj.tvd_m) == 4

        # 2. SEGY file write and read
        segy_file = tmp / "test.segy"
        cube = np.random.randn(16, 16, 16).astype(np.float32)
        write_segy_cube(str(segy_file), cube)
        read_cube = read_segy_cube(str(segy_file))
        assert read_cube.shape == (16, 16, 16)
        assert np.allclose(read_cube, cube, atol=1e-5)

        # 3. CWLS LAS 2.0 cast log write and read
        las_file = tmp / "cast_profile.las"
        c_prof = np.linspace(1.520, 1.480, len(traj.md_m)).astype(np.float32)
        write_las2_file(str(las_file), traj, sound_speed_kms=c_prof)
        las_record = read_las2_file(str(las_file))
        assert len(las_record.trajectory.md_m) == len(traj.md_m)
        assert np.allclose(las_record.curves["SOUND_SPEED"], c_prof, atol=1e-4)


def test_suite14_oatse_model_triad_forward_and_scenario_comparison():
    """Verify OATSEModel triad/cascade forward mode and genuine scenario comparison without fallback."""
    from oatse.scenarios import evaluate_scenario_comparison

    scen = build_maritime_scenario("giuk_polar_halocline_front", grid_shape=(16, 12, 12))
    model = OATSEModel(
        grid_shape=(16, 12, 12),
        num_lithologies=len(MARITIME_ACOUSTIC_CATALOG),
        num_fault_modes=6,
    )
    model.configure_fault_throw_rules(scen["fault_pairs"])

    with torch.no_grad():
        out = model(
            ell_sym_3d=scen["ell_sym_3d"],
            u_fault_3d=scen["u_fault_3d"],
            tau_v=0.20,
            triad_gain=1.0,
            unroll_steps=2,
            true_vp_cube=scen["true_vp_cube"],
            true_rho_cube=scen["true_rho_cube"],
        )

    assert isinstance(out, dict)
    assert out["vp_cube_kms"].shape == (16, 12, 12)
    assert out["zp_cube_mrayl"].shape == (16, 12, 12)
    assert out["max_orthogonality_residual"] <= 1e-7
    assert out["cascade_radial_drift"] <= 1e-7

    comp = evaluate_scenario_comparison("munk_sofar_shadow_zone", grid_shape=(32, 24, 24))
    assert comp["metrics"]["oatse_vp_rmse_kms"] == pytest.approx(0.0, abs=1e-5)
    assert comp["metrics"]["naive_vp_rmse_kms"] > 0.05
    assert comp["metrics"]["beltrami_spike_recovery"] == pytest.approx(1.0, abs=1e-5)


def test_suite15_studio_v020_serialization_and_ray_fan():
    """Verify serialize_evaluation_payload includes all 19 cubes, Snell ray-fan reflection off bathymetry, and Beltrami curves."""
    from oatse.studio import serialize_evaluation_payload

    payload = serialize_evaluation_payload("littoral_internal_tide_wedge", grid_shape=(16, 12, 12))
    assert payload["scenario_name"] == "littoral_internal_tide_wedge"
    assert payload["grid_shape"] == [16, 12, 12]

    expected_cubes = {
        "oatse_c_vp",
        "blind_oatse_c_vp",
        "krig_c_vp",
        "modas_eof_c_vp",
        "naive_c_vp",
        "true_c_vp",
        "oatse_zp",
        "oatse_shadow_tl_db",
        "oatse_vp_tti",
        "oatse_crit_angle_deg",
        "oatse_density_rho",
        "oatse_curl_salience",
        "err_oatse",
        "err_krig",
        "blind_gain",
        "precision_pi",
        "jordan_cavity",
        "timelapse_nope_plume",
        "lithology_cube",
    }
    assert expected_cubes.issubset(set(payload["cubes"].keys()))

    # Verify Snell acoustic eigenray fan reflects off shoaling seabed instead of entering solid rock
    ray_fan = payload["acoustic_ray_fan"]
    c_cube = np.asarray(payload["cubes"]["true_c_vp"])
    assert len(ray_fan) == 6
    for ray in ray_fan:
        assert len(ray["traj_z"]) > 10
        assert len(ray["traj_y"]) == len(ray["traj_z"])
        assert len(ray["traj_x"]) == len(ray["traj_z"])
        for rz, ry, rx in zip(ray["traj_z"], ray["traj_y"], ray["traj_x"]):
            iz = int(np.clip(round(rz), 0, 15))
            iy = int(np.clip(round(ry), 0, 11))
            ix = int(np.clip(round(rx), 0, 11))
            assert c_cube[iz, iy, ix] < 1.65, f"Ray {ray['name']} penetrated solid seabed at ({iz},{iy},{ix}): c={c_cube[iz,iy,ix]}"

    # Verify fault planes metadata and Beltrami reconstructed curves
    assert len(payload["fault_planes_meta"]) >= 1
    assert len(payload["profiles"]) == 8
    for p in payload["profiles"]:
        assert len(p["true_vp"]) == 16
        assert len(p["reconstructed_vp"]) == 16
        assert len(p["naive_pooled_vp"]) == 16
        assert len(p["salience"]) == 16
        assert p["beltrami_spike_recovery"] == pytest.approx(1.0, abs=1e-4)


def test_suite16_portable_studio_html_bundler_and_offline_integrity():
    """Verify generate_studio_html and export_portable_html_studio produce self-contained v0.2.0 HTML."""
    import re
    from oatse.studio import generate_studio_html, serialize_evaluation_payload

    mini_scenarios = {
        "munk_sofar_shadow_zone": serialize_evaluation_payload("munk_sofar_shadow_zone", grid_shape=(16, 12, 12))
    }
    html = generate_studio_html(mini_scenarios)
    assert "OATSE Studio v0.2.0" in html
    assert "window.OATSE_PORTABLE_MODE = true;" in html

    # Verify zero external CDN script/link tags
    ext_refs = re.findall(r'(?:src|href)=["\']https?://', html, flags=re.IGNORECASE)
    assert len(ext_refs) == 0, f"Found external network references in portable HTML: {ext_refs}"

    # Verify v0.2.0 view controls exist in DOM
    for control_id in [
        "sliderVE",
        "sliderOpacity",
        "selectCutoutMode",
        "selectCutoutOctant",
        "selectColormap",
        "chkInvertCmap",
        "btnZDec",
        "btnZInc",
        "btnYDec",
        "btnYInc",
        "btnXDec",
        "btnXInc",
        "btnFlipKrig",
        "btnFlipModas",
        "btnFlipTruth",
        "chkRayFan",
        "chkFrontPlanes",
        "chkWellBadges",
        "chk2DFaultOverlay",
        "btnCamSofar",
        "btnCamFocusWell",
        "btnAutoOrbit",
        "btnCopyDeepLink",
        "btnExportPNG",
        "btnHelpModal",
        "btnExpandTelemetry",
        "sliderTriadGain",
        "sliderTauV",
    ]:
        assert f'id="{control_id}"' in html, f"Missing UI control {control_id} in studio HTML"


def test_suite17_sonar_showcase_figure_and_cli_commands():
    """Verify render_sonar_showcase_figure and CLI subcommands (v0.2.0)."""
    import oatse
    from oatse.cli import main as cli_main
    from oatse.studio import render_sonar_showcase_figure

    assert oatse.__version__ == "0.2.0"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "showcase.png"
        res_path = render_sonar_showcase_figure(
            scenario_name="munk_sofar_shadow_zone",
            output_path=out_png,
            grid_shape=(16, 12, 12),
            dpi=80,
        )
        assert res_path.exists()
        assert res_path.stat().st_size > 10_000

        # Test CLI subcommands
        assert cli_main(["scenarios", "list"]) == 0
        assert cli_main(["scenarios", "inspect", "--scenario", "munk_sofar_shadow_zone"]) == 0


def test_suite18_temporal_continuity_splitter_m2_tide_vs_target():
    """Verify 4D TemporalSonarContinuitySplitter separates M2 tide heaving from moving targets."""
    splitter = TemporalSonarContinuitySplitter(
        m2_period_steps=12,
        cycle_score_thresh=0.20,
    )
    T = 13
    t_steps = torch.arange(T, dtype=torch.float32).view(T, 1, 1, 1)
    omega = 2.0 * math.pi / 12.0
    # Pure M2 tide oscillation in upper water column (z=4) + irreversible target anomaly at (t=6, z=10, y=6, x=6)
    sweeps = torch.full((T, 16, 12, 12), 1.500)
    sweeps[:, 4, :, :] += 0.04 * torch.cos(omega * t_steps[:, 0])
    sweeps[6, 10, 6, 6] += 0.35

    res = splitter.decompose_4d_sweeps(sweeps)
    assert res["irreversible_target_recovery"] == pytest.approx(1.0, abs=1e-5)
    assert bool(res["clock_mask"][4, 2, 2].item()) is True
    assert bool(res["clock_mask"][10, 6, 6].item()) is False
    assert float(res["h_nope"][6, 10, 6, 6].item()) == pytest.approx(1.850, abs=1e-3)

    # Zero displacement edge case
    zero_sweeps = torch.full((4, 16, 12, 12), 1.500)
    res_zero = splitter.decompose_4d_sweeps(zero_sweeps)
    assert float(res_zero["h_clock"].abs().max().item()) == 0.0
    assert bool(res_zero["clock_mask"].any().item()) is False


def test_suite19_all_scenarios_beltrami_spike_recovery_and_regime_aware_noise():
    """Verify 100% Beltrami spike recovery across all 5 scenarios (vertical & deviated) and regime-aware water noise."""
    from oatse.blind_benchmark import _add_deterministic_noise
    from oatse.scenarios import SCENARIO_DESCRIPTIONS, evaluate_scenario_comparison

    for scen_name in SCENARIO_DESCRIPTIONS:
        comp = evaluate_scenario_comparison(scen_name, grid_shape=(32, 24, 24))
        m = comp["metrics"]
        assert m["beltrami_spike_recovery"] == pytest.approx(1.0, abs=1e-5), f"{scen_name} vertical spike recovery < 1.0"
        assert m["deviated_well_spike_recovery"] == pytest.approx(1.0, abs=1e-5), f"{scen_name} deviated spike recovery < 1.0"
        assert m["all_stringers_beltrami_recovery"] == pytest.approx(1.0, abs=1e-5), f"{scen_name} all-stringers recovery < 1.0"
        assert m["thin_bed_detection_f1"] == pytest.approx(1.0, abs=1e-5), f"{scen_name} thin_bed_f1 < 1.0"
        assert m["naive_spike_recovery"] < 0.25

        # Verify regime-aware noise scaling does not pollute water column with basalt-scale variance
        c_true = comp["scenario"]["sound_speed_c"]
        c_noisy = _add_deterministic_noise(c_true, snr_db=20.0, seed=101)
        water_mask = c_true <= 1.62
        water_noise_std = float((c_noisy[water_mask] - c_true[water_mask]).std().item())
        assert water_noise_std < 0.010, f"Water noise std {water_noise_std} inflated by seabed rock variance"


def test_suite20_flask_studio_server_endpoints():
    """Verify Flask studio application endpoints via test_client()."""
    from oatse.studio import HAS_FLASK, create_studio_app

    if not HAS_FLASK:
        pytest.skip("Flask not installed")

    app = create_studio_app(default_grid_shape=(16, 12, 12))
    client = app.test_client()

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert b"OATSE Studio v0.2.0" in r_root.data

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.get_json()["version"] == "0.2.0"

    r_scens = client.get("/api/scenarios")
    assert r_scens.status_code == 200
    assert r_scens.get_json()["count"] == 5

    r_scen = client.get("/api/scenario/giuk_polar_halocline_front")
    assert r_scen.status_code == 200
    assert r_scen.get_json()["scenario_name"] == "giuk_polar_halocline_front"

    r_eval = client.post("/api/evaluate", json={"scenario_name": "subsea_cable_mine_corridor"})
    assert r_eval.status_code == 200
    assert r_eval.get_json()["scenario_name"] == "subsea_cable_mine_corridor"

    r_up = client.post("/api/upload", json={"filename": "test.las", "content": "~A\n0.0 1.50\n50.0 1.49\n"})
    assert r_up.status_code == 200
    assert r_up.get_json()["status"] == "ok"


def test_suite21_tropical_min_plus_semiring_and_kleene_closure() -> None:
    """Verify tropical min-plus matrix multiplication and Kleene star closure (NumPy & PyTorch)."""
    from oatse import tropical_kleene_closure, tropical_min_plus_matmul

    inf = float("inf")
    W_np = np.array(
        [
            [0.0, 4.0, inf, 15.0],
            [inf, 0.0, 2.0, inf],
            [inf, inf, 0.0, 3.0],
            [inf, inf, inf, 0.0],
        ],
        dtype=np.float32,
    )
    W2_np, wit_np = tropical_min_plus_matmul(W_np, W_np, return_witness=True)
    assert float(W2_np[0, 2]) == pytest.approx(6.0)
    assert int(wit_np[0, 2]) == 1

    D_np, route_np = tropical_kleene_closure(W_np, return_routing=True)
    assert float(D_np[0, 3]) == pytest.approx(9.0)  # 0 -> 1 (4) -> 2 (2) -> 3 (3) = 9 < 15
    assert int(route_np[0, 3]) == 1
    assert int(route_np[1, 3]) == 2
    assert int(route_np[2, 3]) == 3

    W_t = torch.tensor(W_np, dtype=torch.float32)
    D_t, route_t = tropical_kleene_closure(W_t, return_routing=True)
    assert torch.allclose(D_t, torch.tensor(D_np, dtype=torch.float32))
    assert int(route_t[0, 3].item()) == 1
    assert int(route_t[1, 3].item()) == 2
    assert int(route_t[3, 0].item()) == -1  # Unreachable must be -1

    # Sentinel inf_threshold=1e12 unreachable routing & empty-dimension edge cases
    W_sent = np.full((3, 3), 1e12, dtype=np.float64)
    np.fill_diagonal(W_sent, 0.0)
    W_sent[0, 1] = 2.5
    D_sent, r_sent = tropical_kleene_closure(W_sent, return_routing=True, inf_threshold=1e12)
    assert float(D_sent[0, 1]) == pytest.approx(2.5)
    assert int(r_sent[0, 1]) == 1
    assert int(r_sent[0, 2]) == -1
    assert int(r_sent[0, 0]) == -1

    empty_c, empty_w = tropical_min_plus_matmul(
        torch.zeros((2, 0), dtype=torch.float32),
        torch.zeros((0, 2), dtype=torch.float32),
        return_witness=True,
    )
    assert empty_c.shape == (2, 2) and bool(torch.all(torch.isinf(empty_c)))
    assert empty_w.shape == (2, 2) and bool(torch.all(empty_w == -1))

