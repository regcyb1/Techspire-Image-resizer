# ⚡ Image Resizer

High-performance local image resizing tool — runs in your browser, processes on your machine.

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
# macOS / Linux:
source venv/bin/activate
# Windows (CMD):
venv\Scripts\activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python resize_app.py

# 5. Open http://localhost:7860
```

---

## 📦 Download for Non-Technical Users (No Installation Required)

You do not need to install Python or use a terminal. Just download the app and double-click it!

1. Go to the **Releases** page (or use the provided download link).
2. Download the version for your computer:
   - **Windows**: `ImageResizer-Windows.exe`
   - **Mac**: `ImageResizer-macOS`
   - **Linux**: `ImageResizer-Linux`
3. Double-click the downloaded file to run it. The app will automatically open in your web browser.

> **Note**: For Mac and Linux, you might need to right-click and allow execution the first time, or run `chmod +x ImageResizer-macOS` in the terminal.

---

## 🛠️ Developer Installation (For PyTorch GPU Support)

If you are a developer and want maximum speed via PyTorch GPU hardware acceleration (NVIDIA CUDA or Apple Metal MPS), follow these steps to install via Python.

```bash
python3 -m venv venv
source venv/bin/activate

# Core + GPU (Apple MPS Metal acceleration)
pip install gradio Pillow torch torchvision

# Optional: 5-10x faster CPU processing
brew install vips
pip install pyvips

python resize_app.py
```

> **GPU**: PyTorch automatically detects Apple MPS. Select **"Apple MPS (Metal)"** in the Compute Device dropdown.

### macOS (Intel)

```bash
python3 -m venv venv
source venv/bin/activate

pip install gradio Pillow torch torchvision
python resize_app.py
```

> **Note**: Intel Macs have no GPU acceleration. PyTorch will run in CPU mode.

### Windows (NVIDIA GPU)

```cmd
python -m venv venv
venv\Scripts\activate

:: Core dependencies
pip install gradio Pillow

:: PyTorch with CUDA 12.1 (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: Optional: fast CPU backend
pip install pyvips

python resize_app.py
```

> **GPU**: Select **"CUDA:0 – [Your GPU Name]"** in the Compute Device dropdown.

### Windows (No NVIDIA GPU)

```cmd
python -m venv venv
venv\Scripts\activate

pip install gradio Pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python resize_app.py
```

### Linux (NVIDIA GPU)

```bash
python3 -m venv venv
source venv/bin/activate

# Core
pip install gradio Pillow

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: fast CPU backend
sudo apt install libvips-dev   # Debian/Ubuntu
sudo dnf install vips-devel     # Fedora/RHEL
pip install pyvips

python resize_app.py
```

### Linux (CPU only)

```bash
python3 -m venv venv
source venv/bin/activate

pip install gradio Pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python resize_app.py
```

---

## Features

- **Batch processing** — handles 1000+ images with parallel workers
- **GPU acceleration** — NVIDIA CUDA and Apple MPS (Metal) via PyTorch
- **Preset resolutions** — 512², 1024², 2048², 4096² or custom
- **PPI/DPI control** — 72, 150, 300, or custom
- **All output is JPG** — PNG, WEBP, TIFF, BMP → `.jpg`
- **Filenames preserved** — `IMG_2034.PNG` → `IMG_2034.jpg`
- **EXIF orientation** — auto-corrected before resize
- **ZIP download** — batch results packaged automatically
- **Auto port recovery** — kills stale processes on port 7860

---

## Performance

| Backend | Device | 100 images (1024²) | 1000 images (1024²) |
|---------|--------|--------------------|--------------------|
| Pillow  | CPU    | ~15s               | ~2.5 min           |
| pyvips  | CPU    | ~3s                | ~30s               |
| PyTorch | Apple MPS | ~5s             | ~50s               |
| PyTorch | CUDA   | ~2s                | ~20s               |

*Benchmarks vary by hardware. CPU parallel workers scale with core count.*

---

## Packaging as Standalone Executable

If you modify the code and want to build your own executable:

```bash
# This script handles downloading PyInstaller and configuring all Gradio assets
python build.py
```

The executable will be placed in the `dist/` folder.
