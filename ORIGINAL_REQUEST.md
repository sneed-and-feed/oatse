# Original User Request

## 2026-10-21T13:19:59Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: 4-worker parallel build team with preliminary arXiv research phase (full team)

Build `oatse` (`v0.1.0`, **O**cean **A**coustic **T**hermocline & **S**onar **E**ngine, `CC0-1.0` Public Domain) from a clean repository into a complete, standalone 3D/4D underwater acoustics, thermocline ray-refraction, and seabed Synthetic Aperture Sonar (SAS) impedance inversion workstation powered by the machine-checked mathematical physics primitives of `OcclusionTriad.lean`, beginning with a preliminary arXiv literature research phase and culminating in a verified git push to `https://github.com/sneed-and-feed/oatse`.

Working directory: `c:\Users\x\Documents\antigravity\oatse`
Integrity mode: development

Reference repositories in workspace:
- `c:\Users\x\Documents\antigravity\g3vm`
- `c:\Users\x\Documents\antigravity\darksnail` (`projectdarksnail`)
- Target GitHub repository: `https://github.com/sneed-and-feed/oatse`

## Requirements

### R0. Preliminary arXiv Literature Research Phase
- Before or alongside core implementation, conduct a preliminary arXiv research phase covering:
  1. Ocean acoustic thermocline ray refraction, Munk (1974) canonical SOFAR profiles, Mackenzie (1981) sound-speed equations, Sonic Layer Depth (SLD) shadow zones, and acoustic transmission loss modeling.
  2. 4D internal-tide ($M_2$) acoustic travel-time fluctuations vs. irreversible target transit and reverberation/bubble washout rejection.
  3. Seabed Synthetic Aperture Sonar (SAS) acoustic impedance inversion ($Z_p = \rho V_p$), sub-bottom Tilted Transversely Isotropic (TTI) sediment elasticity, and Beltrami/topological feature recovery.
- Synthesize relevant arXiv preprints and canonical literature to inform the physical parametrization and cite them in `README.md` and `docs/ARCHITECTURE.md`.

### R1. Architectural Objective, Hardware Invariants & 4-Worker Parallel Topology
Every module must enforce the hardware and mathematical invariants established across `projectdarksnail` (`c:\Users\x\Documents\antigravity\darksnail`) and `g3vm` v0.5.0 (`c:\Users\x\Documents\antigravity\g3vm`):
- **Flight/UUV SWaP-C Memory Envelope (`< 35 MB` Hard Ceiling, `~1.30–1.60 MB` Active Heap):** Evaluate all 3D spectral, multi-bounce ray, $\beta_1$ shadow-zone/gas-pocket cavity enclosures, and $\mathfrak{so}(C)$ triad operators in $Z$-slabs (`chunk_z = 16`) and stream `.segy` volumes via out-of-core Inline-slab `np.memmap` windowing so the entire working set fits inside a 4 MB UUV/glider L2/L3 SRAM cache.
- **Machine-Checked `OcclusionTriad.lean` Runtime Invariants (`model.verify_formal_theorems()`):**
  1. `Theorem 2.1 & 2.2` (`skew_triad_swap_neg`, `occlusion_triad_breaks_symmetric_tie`): Exact skew-symmetry (`0.00e+0` residual) and razor-sharp step-function sound-speed ($c$) and impedance ($Z_p$) jumps across the Sonic Layer Depth (SLD) thermocline, halocline fronts, and the water-to-seabed unconformity.
  2. `Theorem 2.3 & 2.6` (`darksnail_readout_energy_conservation`, `darksnail_multistep_spherical_invariance`): Pointwise tangent orthogonality (`<= 1.0e-8`) and `0.000e+00` radial norm drift across `K in {1, 4, 8, 16}` cascade steps, preventing ray-tomography blowup inside deep acoustic shadow zones.
  3. `Theorem 2.4` (`darksnail_step0_system1_homotopy`): Exact reduction to the baseline Munk (1974) / Mackenzie (1981) stratified ocean-plus-seabed prior (`0.00e+0` residual) when local perturbation `delta_z = 0`.
  4. `Theorem 2.5` (`rope2d_medoid_relative_invariance` & Non-Circular `BeltramiCastShell`): `1.0000` (100.0% clean) exterior-curl + 3-point discrete Morse ($\Delta_- v \cdot \Delta_+ v > 0$) spike recovery of sub-meter seabed acoustic impedance ($Z_p = \rho c$) targets (proud/buried mines, subsea cable taps, boulders, and fine-scale thermocline microstructure sheets) against Equal-Budget ($K = 16$) uniform pooling under 10% acoustic reverberation / bubble washout.
  5. `Theorem 2.7` ($\mathrm{SO}(6)$ Kelvin-Mandel Stiffness Rotor): Exact trace and Frobenius norm conservation (`<= 1.14e-13` residual) across tilted transversely isotropic (TTI) marine turbidite/clay beds and sub-bottom fault scarps.

#### 4-Worker Parallel Topology & Strict File Ownership (No Referee Subagent)

| Agent ID | Role | Owned Files (Exclusive Write Access) | Synchronous Self-Verification Command |
| :--- | :--- | :--- | :--- |
| **`agent-1-core`** | Core Ocean Acoustics, $\mathfrak{so}(C)$ Thermocline Triad, Non-Circular `BeltramiCastShell` & 4D Internal-Tide Splitter | `pyproject.toml`, `LICENSE`, `.gitignore`, `formalization/OcclusionTriad.lean`, `oatse/__init__.py`, `oatse/acoustics.py`, `oatse/engine.py` | `python -c "from oatse.engine import OATSEModel; m = OATSEModel(); print(m.verify_formal_theorems())"` |
| **`agent-2-scenarios-io`** | 5-Scenario Maritime Deck (`01`–`05`), Out-of-Core `.segy` Memmap Streamer, XBT/CTD `.las`/`.dev` I/O & CLI | `oatse/scenarios.py`, `oatse/io.py`, `oatse/cli.py`, `data/noaa_xbt/north_atlantic_sofar.las`, `data/noaa_xbt/uuv_glide2.dev`, `data/noaa_xbt/auv_sas3.dev` | `python -c "from oatse.scenarios import build_all_scenarios; s = build_all_scenarios(); assert len(s) == 5"` |
| **`agent-3-blind-bench`** | Mode A + Mode B Blind Benchmark (`SNR=20dB`), Noise/Jitter Stress Tests & Pytest Suite | `oatse/blind_benchmark.py`, `scripts/evaluate_oatse.py`, `tests/test_oatse.py`, `results/oatse_evaluation_summary.json`, `results/oatse_referee_audit_summary.json` | `python -m pytest tests/test_oatse.py -v && python scripts/evaluate_oatse.py` |
| **`agent-4-studio-docs`** | Zero-Install WebGL 3D Sonar Studio, Math Linter, CI Workflow & Documentation | `oatse/studio.py`, `results/oatse_portable_studio.html`, `scripts/audit_gfm_math.py`, `.github/workflows/ci-and-release.yml`, `README.md`, `docs/ARCHITECTURE.md` | `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` |

### R2. Workstream 1 (`agent-1-core`): `oatse/acoustics.py` & `oatse/engine.py`
1. **`MackenzieMunkAcousticEngine` (`oatse/acoustics.py`):**
   - Implement the 9-term **Mackenzie (1981)** seawater sound-speed equation $c(T, S, z)$ (in $\text{m/s}$ and $\text{km/s}$) coupled with the canonical **Munk (1974)** canonical SOFAR velocity profile:
     $$c(z) = c_0 \left[ 1 + \epsilon_{\mathrm{munk}} \left( \tilde{\eta} - 1 + e^{-\tilde{\eta}} \right) \right], \quad \tilde{\eta} = \frac{2(z - z_{\mathrm{axis}})}{B_{\mathrm{scale}}}$$
   - Compute 3D Snell's-law critical grazing angle $\theta_c(z,y,x) = \arccos\!\bigl(\min(1, c(z,y,x)/c_{\mathrm{SLD}})\bigr)$, multi-bounce surface-duct / bottom-bounce ray quantiles ($1/N_{\mathrm{vox}}$ surgical weighting), **Acoustic Shadow-Zone Exposure & Transmission Loss (`oatse_shadow_tl_db` in $\text{dB}$)**, and **Coupled Water-to-Seabed Acoustic Impedance (`oatse_zp` $= \rho V_p$ in $\text{km/s}\cdot\text{g/cm}^3$)** plus **Sub-Bottom TTI Elastic $V_p(\psi)$ (`oatse_vp_tti`)**.
2. **`SkewSymmetricThermoclineTriad` & $Z$-Slab Engine (`oatse/engine.py`):**
   - Evaluate 3D cubes `(Z=32, Y=24, X=24)` in $Z$-slabs (`chunk_z = 16`, `< 35 MB` heap) using $\mathfrak{so}(C)$ skew-symmetric tie-breaking across sharp thermocline/halocline steps and the water-sediment seafloor interface, 6-way Jordan `cavity_core` ($\beta_1$) isolation for enclosed sub-thermocline quiet targets and shallow seabed gas pockets, and $\mathrm{SO}(6)$ Kelvin-Mandel stiffness rotors in the marine sediment/bedrock layers.
3. **Pure Non-Circular `BeltramiCastShell` ($K = 8 + 8 = 16$ Tokens):**
   - Strictly non-circular Tier-1 spike selection using coarse-fine Beltrami exterior-curl salience, fine-shell magnitude $\vert{}u_{\mathrm{fine}}(z)\vert{}$, and 3-point discrete Morse impulse $\Delta_- v(z)\cdot\Delta_+ v(z) > 0$ (zero access to `spike_anom` or ground truth), paired with a 10%–20% acoustic reverberation / bubble-sweep washout rejection gate.
4. **4D Internal Tide & Bragg Swell Continuity Splitter (`TemporalSonarContinuitySplitter`):**
   - Decompose 4D repeat sonar sweeps into $\mathcal{H}_{\mathrm{clock}} \oplus \mathcal{H}_{\mathrm{nope}}$, quarantining reversible $M_2$ internal-tide thermocline heaving (`continuity_cycle_score > 0`) while recovering `1.0000` (`100.0%`) of irreversible moving sub-thermocline targets or seabed scour/mine placement in $\mathcal{H}_{\mathrm{nope}}$.

### R3. Workstream 2 (`agent-2-scenarios-io`): `oatse/scenarios.py`, `oatse/io.py` & `oatse/cli.py`
Implement the **5-Scenario Maritime & Seabed Benchmark Deck** on `(Z=32, Y=24, X=24)` grids (`18,432` voxels) with 8 vertical/deviated XBT, CTD, UUV glider, and AUV SAS profiles per scenario:
1. **`01` `munk_sofar_shadow_zone`:** Deep-water warm surface mixed duct ($c = 1.538\text{ km/s}$), sharp Sonic Layer Depth (SLD) thermocline step into a cold sub-thermocline shadow zone ($c = 1.462\text{ km/s}$) enclosing a quiet $\beta_1$ target (`cavity_core`), deep SOFAR channel axis, and abyssal pelagic clay/basalt floor.
2. **`02` `giuk_polar_halocline_front`:** Inclined sub-polar oceanic front (North Atlantic / GIUK Gap geometry) where warm saline Atlantic water collides across a tilted step-fault front (`throw = 3 voxels`) with cold, fresh Arctic intermediate water above a faulted glacial-till seabed and high-impedance glacial boulder stringers.
3. **`03` `littoral_internal_tide_wedge`:** Coastal shelf-break wedge with a sharp seasonal pycnocline, shallow sub-bottom free-gas pocket ($V_p = 0.72\text{ km/s}$, $Z_p = 0.994\text{ km/s}\cdot\text{g/cm}^3$ $\beta_1$ enclosure causing severe multi-bounce reverberation), and 4D reversible $M_2$ internal-tide oscillation vs. irreversible target transit.
4. **`04` `subsea_cable_mine_corridor`:** Seabed warfare & offshore wind/telecom route scenario across a sharp water-to-turbidite unconformity (`Z=14`, TTI $(\epsilon, \delta, \gamma) = (0.09, 0.03, 0.11)$), cut by a sub-bottom fault scarp and populated with thin `1–2` voxel proud/partially buried metallic mine and cable-tap impedance spikes ($Z_p = 23.95\text{ km/s}\cdot\text{g/cm}^3$) scanned by a `68°` terrain-following `AUV-SAS3` lateral.
5. **`05` `north_atlantic_xbt_field_doublet` (Real-Profile Benchmark):** Bundled real-parameter NOAA/WOD North Atlantic XBT/CTD sound-speed/density cast (`data/noaa_xbt/north_atlantic_sofar.las`) paired with a `54°` sawtooth `UUV-GLIDE2` thermocline glider and `68°` `AUV-SAS3` seabed survey (`< 200 KB` total committed data).
6. **Out-of-Core `.segy` & `.las` I/O (`oatse/io.py` & `oatse/cli.py`):**
   - Implement `stream_segy_volume()` using Inline-slab `np.memmap` windowing (`< 35 MB` peak heap on a `64 MB` `400x400x100` `.segy` file) with 3D Beltrami exterior-curl downsampling (`--roi`, `--max-grid`), plus CWLS `.las` 2.0 (`SOUND_SPEED`, `RHO`, `ZP_SONAR`, `TL_DB`) and minimum-curvature `.dev` UUV trajectory parsers.

### R4. Workstream 3 (`agent-3-blind-bench`): `oatse/blind_benchmark.py`, `scripts/evaluate_oatse.py` & `tests/test_oatse.py`
Build the complete dual-mode evaluation and stress-test suite:
1. **Mode A (Oracle Operator Verification) & Mode B (Blind Sparse-Cast & Noisy Sonar Inversion at `SNR = 20 dB`, Zero Target Leakage):**
   - In Mode B, construct `ell_sym_blind`, `u_front_blind`, and travel-time residuals strictly from training XBT/CTD casts `W_train` + $3\times 3\times 3$ Fresnel-smoothed noisy acoustic migration velocity at `SNR = 20 dB` with **zero access to `true_litho_cube` or held-out casts/UUV trajectories**.
   - Compare `NaiveBox`, **`TunedAnisotropicKriging`** (inner LOWO CV-tuned 3D Matérn-3/2), **`FrontPartitionedKriging`**, and **`BlindOATSE`** across:
     - Leave-One-Cast-Out (`LOWO`) vertical XBT/CTD casts,
     - Unseen 3D deviated UUV glider / AUV SAS trajectories,
     - Whole-Cube $c / V_p$ RMSE and Seabed $Z_p$ RMSE, and
     - Symmetric **Bounded Edge Gradient Fidelity** (`oatse_edge_gradient_fidelity` $\in [0\%, 100\%]$).
2. **Equal-Budget ($K=16$) `BeltramiCastShell`, SNR Sweep ($\infty, 20, 10, 5\text{ dB}$) & Thermocline-Pick Jitter ($\Delta z_{\mathrm{jitter}} \in \{0, 1, 2, 3\}$ Voxels) Stress Tests:**
   - Benchmark Peak / All-Stringers / Thin-Target F1 against Equal-Budget ($K=16$) and Coarse (`Z//8`) pooling across Clean, `20 dB`, `15 dB + 10% reverberation washout`, and `10 dB + 20% washout`.
   - Document honest regime boundaries when front/thermocline picks are shifted by $\ge 3$ voxels.
3. **Pytest Suite (`tests/test_oatse.py`):**
   - Assert all `OcclusionTriad.lean` runtime theorems (`Thm 2.1–2.7`), Mode A & Mode B accuracy wins, `chunk_z = 8 / 16 / 32` bit-identity (`< 1e-6`), and `tracemalloc` peak heap `< 35 MB` on both 3D evaluation and `64 MB` `.segy` Inline-slab streaming.

### R5. Workstream 4 (`agent-4-studio-docs`): `oatse/studio.py`, `results/oatse_portable_studio.html`, Docs & Git Push
1. **Zero-Install HTML5/WebGL 3D Sonar & Thermocline Studio (`results/oatse_portable_studio.html`):**
   - Single-file dark-mode Three.js / WebGL workstation supporting `file://` and GitHack CDN deep-linking (`?scenario=munk_sofar_shadow_zone&field=oatse_shadow_tl_db`) across all 5 scenarios (`01`–`05`).
   - 3D chair-cut block viewer with selectable fields: `Sound Speed c / Vp (km/s)`, `Seabed Acoustic Impedance Zp`, `Shadow-Zone Transmission Loss TL (dB)`, `Sub-Bottom TTI Vp(psi)`, and `Beltrami Exterior Curl Salience`.
   - Render 3D UUV glider (`UUV-GLIDE2`) and AUV SAS (`AUV-SAS3`) trajectory tubes with locked Beltrami impedance/microstructure spikes (yellow rings), rejected reverberation washouts (purple squares), and a client-side **`.las` / `.dev` / `.segy` `ArrayBuffer` Drag-and-Drop Inspector**.
2. **Documentation & Math Audit (`README.md`, `docs/ARCHITECTURE.md`):**
   - Document the mathematical isomorphism between `projectdarksnail` (`OcclusionTriad.lean`), `g3vm`, and `oatse`, incorporate citations from the preliminary arXiv research phase, include the full Mode A and Mode B 5-scenario tables, and verify `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` passes with `0` errors.
3. **Git Initialization & Push (`https://github.com/sneed-and-feed/oatse`):**
   - Initialize the git repository in `c:\Users\x\Documents\antigravity\oatse`, stage all project files, commit `v0.1.0`, and push to `https://github.com/sneed-and-feed/oatse`.

## Acceptance Criteria

### Formal & Runtime Verification
- [ ] `python -c "from oatse.engine import OATSEModel; m = OATSEModel(); print(m.verify_formal_theorems())"` passes all Theorems 2.1–2.7 (`0.00e+0` skew-symmetry residual, `0.000e+00` multistep radial drift, `<= 1.0e-8` tangent orthogonality, `0.00e+0` `delta_z=0` homotopy residual, `1.0000` clean spike recovery under 10% washout, `<= 1.14e-13` SO(6) rotor conservation).
- [ ] `python -c "from oatse.scenarios import build_all_scenarios; s = build_all_scenarios(); assert len(s) == 5"` succeeds with all 5 scenarios (`01`–`05`) and `< 200 KB` committed data in `data/noaa_xbt/`.

### Test Suite, Benchmarks, Studio & Git Delivery
- [ ] `python -m pytest tests/test_oatse.py -v` passes 100% with `0` warnings, `0.000e+00` multistep radial drift, `<= 1.0e-8` tangent orthogonality, and `< 35 MB` (`~1.5 MB` engine / `< 35 MB` `64 MB` `.segy` memmap) peak heap.
- [ ] `python scripts/evaluate_oatse.py` generates `results/oatse_evaluation_summary.json` and `results/oatse_referee_audit_summary.json` across all 5 scenarios (`01`–`05`) in both Mode A and Mode B (`SNR = 20 dB`).
- [ ] `results/oatse_portable_studio.html` opens directly in any browser with zero external network calls, rendering all 5 oceanic/seabed scenarios, UUV trajectories, and client-side `.las`/`.segy` dropzone inspection.
- [ ] `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` passes with `0` issues.
- [ ] Repository `c:\Users\x\Documents\antigravity\oatse` is committed and pushed to `https://github.com/sneed-and-feed/oatse`.


## 2026-10-21T13:43:28Z

## USER ADDENDUM: Official Community Baseline Benchmarks (Integrate Immediately Without Waiting)

The user has requested expanding the benchmark suite in `v0.1.0` immediately with an **Official Community Baseline Benchmarks Addendum**. Please route this immediately to your Orchestrator (`da12ab6a-1904-4f71-acd5-6adf569780f0`) and the relevant workers (`agent-3-blind-bench` and `agent-4-studio-docs`) while preserving strict file ownership:

### 1. For `agent-3-blind-bench` (`oatse/blind_benchmark.py`, `scripts/evaluate_oatse.py`, `tests/test_oatse.py`, `results/*.json`):
1. **Operational `EOFAnisotropicKriging` (`MODAS` / `sEOF-r` Baseline)**:
   - Implement `EOFAnisotropicKriging` in `oatse/blind_benchmark.py`: extract Empirical Orthogonal Functions (SVD/EOF vertical modes) strictly from training XBT/CTD casts `W_train` (zero target leakage) and interpolate the EOF mode coefficients across horizontal $(y, x)$ using 2D Matérn-3/2 Objective Analysis / Kriging (mirroring the operational U.S. Navy MODAS / sEOF-r 3D sound-speed reconstruction method, Fox et al. 2002).
   - Include `EOFAnisotropicKriging` as a baseline comparator alongside `NaiveBox`, `TunedAnisotropicKriging`, `FrontPartitionedKriging`, and `BlindOATSE` in the Mode B evaluation and JSON summaries (`results/oatse_evaluation_summary.json` and `results/oatse_referee_audit_summary.json`).
2. **Official Community Benchmarks Suite (`run_official_community_benchmarks()` in `oatse/blind_benchmark.py`)**:
   - **ASA / OALIB Canonical Pekeris Wedge Benchmark (Jensen & Ferla 1990, *JASA* 87)**:
     - Standard ASA shallow-water wedge benchmark ($200\text{ m} \to 0\text{ m}$ depth over $4\text{ km}$ range at $2.86^\circ$ slope; homogeneous water layer $c_w = 1.500\text{ km/s}$, $\rho_w = 1.00\text{ g/cm}^3$; penetrable fluid/elastic sediment bottom $c_b = 1.700\text{ km/s}$, $\rho_b = 1.50\text{ g/cm}^3$, $Z_p = 1.50 \to 2.55\text{ km/s}\cdot\text{g/cm}^3$).
     - Verify exact Snell critical grazing angle $\theta_c = \arccos(1500/1700) \approx 28.072487^\circ$ (`0.489957 rad`), upslope modal cutoff / shadow transmission loss progression, and sharp inclined seabed $c / Z_p$ recovery (`BlindOATSE` vs. `TunedAnisotropicKriging` and `EOFAnisotropicKriging`).
   - **ONR / NRL Geoacoustic Inversion Workshop Blind Test Case (Chapman & Tolstoy 1998; Chapman et al. 2003, *IEEE J. Oceanic Eng.*)**:
     - Evaluate blind geoacoustic inversion (`Case A/C` sediment layer over high-impedance basement + thin anomalous sub-bottom layer) comparing `BlindOATSE` vs. `EOFAnisotropicKriging` and `TunedAnisotropicKriging` on seabed $V_p$, $Z_p$, and Edge Gradient Fidelity.
   - **OALIB Munk (1974) / BELLHOP WKB Ray-Tube & Thorp Attenuation Reference Verification**:
     - Verify canonical Munk SOFAR axial ray loop length $\Lambda = \pi B_{\mathrm{scale}} \epsilon_{\mathrm{munk}}^{-1/2} \approx 47.663\text{ km}$ ($B_{\mathrm{scale}} = 1.3\text{ km}, \epsilon_{\mathrm{munk}} = 0.00737$) and validate `oatse_shadow_tl_db` against the analytical WKB ray-tube divergence + Thorp attenuation reference curve.
   - Include the output of `run_official_community_benchmarks()` in `results/oatse_evaluation_summary.json` and `results/oatse_referee_audit_summary.json`, and add unit tests in `tests/test_oatse.py` verifying both `EOFAnisotropicKriging` and `run_official_community_benchmarks()`.

### 2. For `agent-4-studio-docs` (`README.md`, `docs/ARCHITECTURE.md`):
- Add a dedicated section/table documenting the **Official Community Baseline Benchmarks Addendum** (`EOFAnisotropicKriging` Navy MODAS baseline, ASA 1990 Pekeris Wedge, ONR/NRL Geoacoustic Inversion Workshop calibration, and OALIB Munk/WKB reference) in `README.md` and `docs/ARCHITECTURE.md`, ensuring `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` continues to pass with `0` errors.
