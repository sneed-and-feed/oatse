#!/usr/bin/env python3
"""
scripts/capture_studio_screenshots.py — Export Portable OATSE Studio v0.2.0 HTML,
render publication-grade Matplotlib showcase figures, and capture 1920x1080 headless
Edge/Chrome screenshots across scenarios, acoustic fields, and view controls.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oatse.studio import export_portable_html_studio, render_sonar_showcase_figure


def find_browser_executable() -> str:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge") or "",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("Could not locate headless Edge or Chrome executable.")


def capture_screenshot(browser_exe: str, url: str, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1920,1080",
        "--virtual-time-budget=2500",
        f"--screenshot={output_png.resolve()}",
        url,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[+] Captured studio screenshot: {output_png} ({output_png.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    studio_dir = ROOT / "studio"
    results_dir = ROOT / "results"
    docs_dir = ROOT / "docs"
    studio_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export portable HTML studio
    html_path = export_portable_html_studio(studio_dir / "oatse_portable_studio.html")
    print(f"[+] Exported portable HTML studio: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")

    # 2. Render Matplotlib showcase figure
    showcase_path = render_sonar_showcase_figure(
        scenario_name="munk_sofar_shadow_zone",
        output_path=results_dir / "oatse_sonar_showcase.png",
    )
    print(f"[+] Rendered Matplotlib showcase figure: {showcase_path}")

    # 3. Capture headless browser screenshots across scenarios & view controls
    browser = find_browser_executable()
    base_uri = html_path.resolve().as_uri()

    shots = [
        (
            f"{base_uri}?scenario=munk_sofar_shadow_zone&field=oatse_c_vp&well=4&sliceZ=16&sliceY=12&sliceX=12&ve=1.25&cutout=chair",
            docs_dir / "oatse_studio_preview.png",
        ),
        (
            f"{base_uri}?scenario=munk_sofar_shadow_zone&field=oatse_c_vp&well=4&sliceZ=16&sliceY=12&sliceX=12&ve=1.25&cutout=chair",
            results_dir / "oatse_studio_v020_munk_sofar.png",
        ),
        (
            f"{base_uri}?scenario=giuk_polar_halocline_front&field=blind_oatse_c_vp&well=5&sliceZ=15&sliceY=12&sliceX=11&yaw=-0.82&pitch=0.44&ve=1.35",
            results_dir / "oatse_studio_v020_giuk_polar_front.png",
        ),
        (
            f"{base_uri}?scenario=littoral_internal_tide_wedge&field=oatse_zp&well=6&sliceZ=14&sliceY=11&sliceX=8&ve=1.45&cutout=chair",
            results_dir / "oatse_studio_v020_littoral_zp.png",
        ),
        (
            f"{base_uri}?scenario=subsea_cable_mine_corridor&field=oatse_shadow_tl_db&well=7&sliceZ=14&sliceY=11&sliceX=11&ve=1.30&cutout=l_section",
            results_dir / "oatse_studio_v020_cable_mine_tl.png",
        ),
        (
            f"{base_uri}?scenario=north_atlantic_xbt_field_doublet&field=oatse_curl_salience&well=0&sliceZ=10&sliceY=11&sliceX=11&ve=1.25",
            results_dir / "oatse_studio_v020_north_atlantic_curl.png",
        ),
    ]

    for url, out_png in shots:
        capture_screenshot(browser, url, out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
