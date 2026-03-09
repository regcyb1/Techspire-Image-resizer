#!/usr/bin/env python3
"""
High-Performance Local Image Resizer
=====================================
A single-file Gradio web application for bulk image resizing.

Run:  python resize_app.py
Open: http://localhost:7860

Features:
  - Resize images to preset or custom dimensions
  - Set PPI/DPI metadata
  - Convert any image format → JPG
  - Parallel batch processing (1000+ images)
  - pyvips (fast) with Pillow fallback
  - Auto EXIF orientation correction
  - ZIP download for batch results
"""

import os
import sys
import time
import shutil
import zipfile
import tempfile
import logging
import pathlib
import struct
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("resizer")

# ---------------------------------------------------------------------------
# Dependency checks & backend selection
# ---------------------------------------------------------------------------
USE_PYVIPS = False
try:
    import pyvips
    USE_PYVIPS = True
    log.info("pyvips %s detected – using high-speed backend.", pyvips.__version__)
except ImportError:
    log.info("pyvips not available – falling back to Pillow.")

try:
    from PIL import Image, ExifTags, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle slightly corrupt files
    log.info("Pillow %s loaded.", Image.__version__)
except ImportError:
    if not USE_PYVIPS:
        sys.exit("ERROR: Neither pyvips nor Pillow is installed. "
                 "Install at least one:\n  pip install Pillow\n  pip install pyvips")

# ---------------------------------------------------------------------------
# GPU / device detection — builds a list of available compute devices
# ---------------------------------------------------------------------------
HAS_TORCH = False
AVAILABLE_DEVICES = ["CPU (Pillow/pyvips)"]
try:
    import torch
    import torch.nn.functional as F  # noqa: N812 – used for GPU resize
    HAS_TORCH = True
    AVAILABLE_DEVICES.append("CPU (PyTorch)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            AVAILABLE_DEVICES.append(f"CUDA:{i} – {torch.cuda.get_device_name(i)}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        AVAILABLE_DEVICES.append("Apple MPS (Metal)")
    log.info("PyTorch %s loaded. Devices: %s", torch.__version__, AVAILABLE_DEVICES)
except ImportError:
    log.info("PyTorch not installed – GPU acceleration unavailable.")

GPU_INFO = ", ".join(AVAILABLE_DEVICES)
log.info("Available devices: %s", GPU_INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"}
MAX_WORKERS = min(os.cpu_count() or 4, 16)  # Cap at 16 to avoid thrashing

PRESET_RESOLUTIONS = {
    "512 × 512": (512, 512),
    "1024 × 1024": (1024, 1024),
    "2048 × 2048": (2048, 2048),
    "4096 × 4096": (4096, 4096),
    "Custom": None,
}

PRESET_PPI = {
    "72": 72,
    "150": 150,
    "300": 300,
    "Custom": None,
}

# ---------------------------------------------------------------------------
# EXIF orientation helpers (Pillow path)
# ---------------------------------------------------------------------------
def _apply_exif_orientation_pillow(img: "Image.Image") -> "Image.Image":
    """Transpose a Pillow image according to its EXIF Orientation tag."""
    try:
        exif = img.getexif()
        orientation = exif.get(0x0112)  # Orientation tag
        if orientation is None:
            return img
        transforms = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }
        if orientation in transforms:
            img = img.transpose(transforms[orientation])
    except Exception:
        pass
    return img


# ---------------------------------------------------------------------------
# Core resize function (runs in worker process)
# ---------------------------------------------------------------------------
def _resize_single_image(
    input_path: str,
    output_dir: str,
    width: int,
    height: int,
    maintain_aspect: bool,
    ppi: int,
    quality: int,
    use_pyvips: bool,
    device: str = "CPU (Pillow/pyvips)",
) -> dict:
    """
    Resize a single image file and save as JPG.

    Returns a dict with keys: ok (bool), src (str), dst (str), error (str|None).
    """
    src = Path(input_path)
    stem = src.stem
    dst = Path(output_dir) / f"{stem}.jpg"

    try:
        # Route to the appropriate backend
        if device != "CPU (Pillow/pyvips)":
            return _resize_torch_gpu(src, dst, width, height, maintain_aspect, ppi, quality, device)
        elif use_pyvips:
            return _resize_pyvips(src, dst, width, height, maintain_aspect, ppi, quality)
        else:
            return _resize_pillow(src, dst, width, height, maintain_aspect, ppi, quality)
    except Exception as exc:
        return {"ok": False, "src": str(src), "dst": str(dst), "error": str(exc)}


def _resize_pyvips(src, dst, w, h, maintain_aspect, ppi, quality):
    """Resize using pyvips (extremely fast, low memory)."""
    img = pyvips.Image.new_from_file(str(src), access="sequential")

    # Auto-rotate based on EXIF
    img = img.autorot()

    # Calculate scale factors
    if maintain_aspect:
        scale = min(w / img.width, h / img.height)
        img = img.resize(scale)
    else:
        h_scale = w / img.width
        v_scale = h / img.height
        img = img.resize(h_scale, vscale=v_scale)

    # Flatten alpha channel to white background
    if img.hasalpha():
        img = img.flatten(background=[255, 255, 255])

    # Convert to sRGB if needed
    if img.interpretation != "srgb":
        img = img.colourspace("srgb")

    # Set PPI in JFIF metadata
    img.jpegsave(
        str(dst),
        Q=quality,
        optimize_coding=True,
        interlace=True,  # progressive
        strip=True,
    )

    # Patch DPI into the saved JPEG via JFIF header (pyvips strip removes it)
    _patch_jpeg_dpi(str(dst), ppi)

    return {"ok": True, "src": str(src), "dst": str(dst), "error": None}


def _patch_jpeg_dpi(filepath: str, ppi: int):
    """Patch the JFIF APP0 header to set DPI on an existing JPEG file."""
    try:
        with open(filepath, "r+b") as f:
            data = f.read(20)
            # Look for JFIF marker: FF D8 FF E0 xx xx 'JFIF\x00'
            if data[0:2] == b'\xff\xd8' and data[2:4] == b'\xff\xe0':
                jfif_start = 4
                length = struct.unpack(">H", data[4:6])[0]
                if data[6:11] == b'JFIF\x00':
                    # Units byte at offset 11 (1 = DPI)
                    f.seek(11)
                    f.write(b'\x01')
                    # X density at offset 12-13, Y density at offset 14-15 (big-endian)
                    f.seek(12)
                    f.write(struct.pack(">HH", ppi, ppi))
    except Exception:
        pass  # Non-critical: DPI metadata is best-effort


def _resize_pillow(src, dst, w, h, maintain_aspect, ppi, quality):
    """Resize using Pillow (universal fallback)."""
    img = Image.open(str(src))

    # EXIF orientation correction
    img = _apply_exif_orientation_pillow(img)

    # Convert to RGB (flatten alpha, handle palette/grayscale)
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    if maintain_aspect:
        img.thumbnail((w, h), Image.LANCZOS)
    else:
        img = img.resize((w, h), Image.LANCZOS)

    # Save JPEG
    img.save(
        str(dst),
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        dpi=(ppi, ppi),
    )

    return {"ok": True, "src": str(src), "dst": str(dst), "error": None}


def _resize_torch_gpu(src, dst, w, h, maintain_aspect, ppi, quality, device_label):
    """Resize using PyTorch on GPU (CUDA or Apple MPS) or CPU."""
    import torch
    import torch.nn.functional as F

    # Determine torch device string from the UI label
    if "CUDA" in device_label:
        # e.g. "CUDA:0 – GeForce RTX 3090" → "cuda:0"
        dev = "cuda:" + device_label.split(":")[1].split(" ")[0]
    elif "MPS" in device_label:
        dev = "mps"
    else:
        dev = "cpu"

    # Load with Pillow (handles format diversity + EXIF)
    img = Image.open(str(src))
    img = _apply_exif_orientation_pillow(img)

    # Convert to RGB
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Calculate target size
    if maintain_aspect:
        scale = min(w / img.width, h / img.height)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
    else:
        new_w, new_h = w, h

    # Convert to tensor → GPU → resize → back to CPU
    import numpy as np
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tensor = tensor.to(dev)

    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Move back to CPU and convert to PIL
    result = resized.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    result_img = Image.fromarray((result * 255).astype(np.uint8), "RGB")

    # Save as JPEG
    result_img.save(
        str(dst),
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        dpi=(ppi, ppi),
    )

    return {"ok": True, "src": str(src), "dst": str(dst), "error": None}


# ---------------------------------------------------------------------------
# Batch processing orchestrator
# ---------------------------------------------------------------------------
def process_images(
    files,
    resolution_preset: str,
    custom_width: int,
    custom_height: int,
    maintain_aspect: bool,
    ppi_preset: str,
    custom_ppi: int,
    quality: int,
    device: str = "CPU (Pillow/pyvips)",
    progress=None,
):
    """
    Process a list of uploaded files.

    Returns (output_path, status_message).
    output_path is either a single .jpg or a .zip file path.
    """
    if not files:
        return None, "⚠️ No files uploaded."

    # --- Resolve parameters ---
    if resolution_preset == "Custom":
        w, h = int(custom_width), int(custom_height)
    else:
        w, h = PRESET_RESOLUTIONS[resolution_preset]

    if w <= 0 or h <= 0:
        return None, "⚠️ Width and height must be positive integers."

    if ppi_preset == "Custom":
        ppi = int(custom_ppi)
    else:
        ppi = PRESET_PPI[ppi_preset]

    if ppi <= 0:
        ppi = 300

    quality = max(70, min(100, int(quality)))

    # --- Filter valid image files ---
    valid_files = []
    for f in files:
        fp = Path(f) if isinstance(f, str) else Path(f.name)
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            valid_files.append(str(fp))
        else:
            log.warning("Skipping unsupported file: %s", fp.name)

    if not valid_files:
        return None, "⚠️ No supported image files found."

    total = len(valid_files)
    log.info("Processing %d images → %dx%d @ %d PPI, quality %d", total, w, h, ppi, quality)

    # --- Create temp output directory ---
    output_dir = tempfile.mkdtemp(prefix="resizer_out_")

    # --- Process images in parallel ---
    t0 = time.time()
    results = []
    errors = []
    completed = 0

    # Use slightly fewer workers than CPUs to keep system responsive
    workers = max(1, min(MAX_WORKERS, total))

    # GPU processing uses ThreadPoolExecutor (GPU ops release GIL) or serial
    # CPU processing uses ProcessPoolExecutor for true parallelism
    use_gpu = device != "CPU (Pillow/pyvips)"

    if use_gpu:
        # GPU: process sequentially to avoid GPU memory contention
        from concurrent.futures import ThreadPoolExecutor
        PoolClass = ThreadPoolExecutor
        workers = 1  # GPU serialised
    else:
        PoolClass = ProcessPoolExecutor

    with PoolClass(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _resize_single_image,
                fp, output_dir, w, h, maintain_aspect, ppi, quality, USE_PYVIPS, device,
            ): fp
            for fp in valid_files
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not result["ok"]:
                errors.append(result)
            completed += 1
            if progress is not None:
                progress(completed / total, desc=f"Processed {completed}/{total}")

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["ok"])
    err_count = len(errors)

    # --- Status summary ---
    speed = ok_count / elapsed if elapsed > 0 else 0
    status_lines = [
        f"✅ **{ok_count}/{total}** images processed in **{elapsed:.1f}s** ({speed:.1f} img/s)",
        f"📐 Resolution: {w} × {h} | 🔘 PPI: {ppi} | 🎨 Quality: {quality}",
        f"⚙️ Workers: {workers} | Device: {device} | Backend: {'pyvips' if USE_PYVIPS and not use_gpu else 'PyTorch GPU' if use_gpu else 'Pillow'}",
    ]
    if err_count:
        status_lines.append(f"⚠️ {err_count} error(s):")
        for e in errors[:10]:
            status_lines.append(f"  • `{Path(e['src']).name}`: {e['error']}")
        if err_count > 10:
            status_lines.append(f"  … and {err_count - 10} more")

    status_msg = "\n".join(status_lines)

    # --- Package output ---
    output_files = sorted(Path(output_dir).glob("*.jpg"))

    if len(output_files) == 0:
        return None, status_msg + "\n\n⚠️ No output files generated."

    if len(output_files) == 1:
        # Single file: return it directly
        return str(output_files[0]), status_msg

    # Multiple files: create ZIP
    zip_path = os.path.join(tempfile.gettempdir(), "resized_images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in output_files:
            zf.write(fp, fp.name)

    # Clean up temp dir (ZIP has the files now)
    shutil.rmtree(output_dir, ignore_errors=True)

    return zip_path, status_msg


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    """Construct and return the Gradio Blocks interface."""
    try:
        import gradio as gr
    except ImportError:
        sys.exit("ERROR: Gradio is not installed.\n  pip install gradio")

    # --- Custom CSS for a polished dark theme ---
    custom_css = """
    /* Global */
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
    }
    #title-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    #subtitle-text {
        text-align: center;
        opacity: 0.65;
        margin-top: 0;
        font-size: 0.95rem;
    }
    .settings-group {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
    }
    #process-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700;
        font-size: 1.1rem;
        border: none !important;
        border-radius: 10px;
        padding: 12px 0;
    }
    #process-btn:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    """

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Image Resizer") as app:
        # --- Header ---
        gr.HTML("<h1 id='title-text'>⚡ Image Resizer</h1>")
        gr.HTML(f"<p id='subtitle-text'>High-performance bulk image processing &nbsp;•&nbsp; Devices: <b>{GPU_INFO}</b></p>")

        with gr.Row():
            # ---- Left column: Settings ----
            with gr.Column(scale=1):
                # Upload
                with gr.Group(elem_classes="settings-group"):
                    gr.Markdown("### 📁 Upload Images")
                    file_input = gr.Files(
                        label="Drop images here or click to browse",
                        file_types=["image"],
                        file_count="multiple",
                    )

                # Resize settings
                with gr.Group(elem_classes="settings-group"):
                    gr.Markdown("### 📐 Resize Settings")
                    resolution_preset = gr.Dropdown(
                        choices=list(PRESET_RESOLUTIONS.keys()),
                        value="1024 × 1024",
                        label="Preset Resolution",
                    )
                    with gr.Row():
                        custom_width = gr.Number(
                            value=1024, label="Width (px)",
                            precision=0, minimum=1, maximum=16384,
                            interactive=False,
                        )
                        custom_height = gr.Number(
                            value=1024, label="Height (px)",
                            precision=0, minimum=1, maximum=16384,
                            interactive=False,
                        )
                    maintain_aspect = gr.Checkbox(
                        value=True,
                        label="Maintain aspect ratio (fit within dimensions)",
                    )

                # PPI settings
                with gr.Group(elem_classes="settings-group"):
                    gr.Markdown("### 🔘 PPI / DPI")
                    ppi_preset = gr.Radio(
                        choices=list(PRESET_PPI.keys()),
                        value="300",
                        label="PPI Preset",
                    )
                    custom_ppi = gr.Number(
                        value=300, label="Custom PPI",
                        precision=0, minimum=1, maximum=2400,
                        interactive=False,
                    )

                # Quality settings
                with gr.Group(elem_classes="settings-group"):
                    gr.Markdown("### 🎨 JPEG Quality")
                    quality = gr.Slider(
                        minimum=70, maximum=100, step=1, value=90,
                        label="Quality (70–100)",
                    )

                # Compute device selector
                with gr.Group(elem_classes="settings-group"):
                    gr.Markdown("### 🖥️ Compute Device")
                    device_selector = gr.Dropdown(
                        choices=AVAILABLE_DEVICES,
                        value=AVAILABLE_DEVICES[0],
                        label="Device",
                        info="Select GPU for hardware acceleration (requires PyTorch)",
                    )
                    gr.Markdown(
                        "<small>Output: **JPG** &nbsp;•&nbsp; optimize: ✅ &nbsp;•&nbsp; progressive: ✅</small>",
                    )

            # ---- Right column: Action & Results ----
            with gr.Column(scale=1):
                process_btn = gr.Button(
                    "⚡ Start Processing",
                    variant="primary",
                    elem_id="process-btn",
                    size="lg",
                )
                status_output = gr.Markdown(
                    value="*Upload images and click **Start Processing** to begin.*",
                    label="Status",
                )
                file_output = gr.File(
                    label="📥 Download Result",
                    interactive=False,
                )

        # --- Interaction logic ---

        def _on_resolution_change(preset):
            """Enable/disable custom width/height based on preset selection."""
            if preset == "Custom":
                return gr.update(interactive=True), gr.update(interactive=True)
            else:
                w, h = PRESET_RESOLUTIONS[preset]
                return gr.update(value=w, interactive=False), gr.update(value=h, interactive=False)

        resolution_preset.change(
            fn=_on_resolution_change,
            inputs=[resolution_preset],
            outputs=[custom_width, custom_height],
        )

        def _on_ppi_change(preset):
            """Enable/disable custom PPI based on preset selection."""
            if preset == "Custom":
                return gr.update(interactive=True)
            else:
                return gr.update(value=PRESET_PPI[preset], interactive=False)

        ppi_preset.change(
            fn=_on_ppi_change,
            inputs=[ppi_preset],
            outputs=[custom_ppi],
        )

        def _run_processing(
            files, res_preset, c_width, c_height, aspect, ppi_p, c_ppi, qual, dev, progress=gr.Progress()
        ):
            """Main processing handler triggered by the button."""
            output_path, status = process_images(
                files, res_preset, c_width, c_height, aspect, ppi_p, c_ppi, qual, dev, progress
            )
            return status, output_path

        process_btn.click(
            fn=_run_processing,
            inputs=[
                file_input, resolution_preset,
                custom_width, custom_height, maintain_aspect,
                ppi_preset, custom_ppi, quality, device_selector,
            ],
            outputs=[status_output, file_output],
        )

    return app, theme, custom_css


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _kill_port(port: int):
    """Kill any process occupying the given port (cross-platform)."""
    import subprocess
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                f"netstat -ano | findstr :{port}", shell=True, text=True
            )
            for line in out.strip().splitlines():
                pid = line.strip().split()[-1]
                if pid.isdigit():
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
        else:
            subprocess.call(f"lsof -ti :{port} | xargs kill -9 2>/dev/null", shell=True)
        log.info("Cleared port %d.", port)
    except Exception:
        pass  # Nothing to kill


def main():
    log.info("=" * 60)
    log.info("  Image Resizer — Starting up")
    log.info("  Backend : %s", "pyvips" if USE_PYVIPS else "Pillow")
    log.info("  Devices : %s", GPU_INFO)
    log.info("  Workers : %d", MAX_WORKERS)
    log.info("=" * 60)

    # Free port 7860 if a stale process is using it
    _kill_port(7860)

    app, theme, css = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
