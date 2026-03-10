import os
import sys
import subprocess
import shutil

def main():
    print("="*50)
    print(" Building Image Resizer Executable ")
    print("="*50)

    # 1. Ensure pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. Determine OS name for the executable output
    if sys.platform == "win32":
        os_name = "Windows"
        exe_ext = ".exe"
    elif sys.platform == "darwin":
        os_name = "macOS"
        exe_ext = ""
    else:
        os_name = "Linux"
        exe_ext = ""

    binary_name = f"ImageResizer-{os_name}{exe_ext}"

    # 3. Clean previous builds
    for d in ["build", "dist"]:
        if os.path.exists(d):
            print(f"Cleaning {d}/ directory...")
            shutil.rmtree(d)

    # 4. Run PyInstaller
    print(f"\nRunning PyInstaller to build {binary_name}...")
    
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed", # Don't open a console window on launch if possible (mac/win)
        f"--name={binary_name}",
        "--clean",
        # Gradio needs its assets and templates
        "--collect-all=gradio",
        "--collect-data=gradio_client",
        # Pillow needs to be explicitly imported sometimes
        "--hidden-import=PIL",
        "--hidden-import=PIL._tkinter_finder",
        "resize_app.py"
    ]
    
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "="*50)
        print("[SUCCESS] Build Successful!")
        print(f"Your standalone executable is located at: dist/{binary_name}")
        print("You can distribute this single file to other users on the same OS.")
        print("="*50)
    else:
        print("\n[ERROR] Build failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
