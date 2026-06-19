# PyInstaller spec — run from repo root:
#   pyinstaller docker/regr_fail_bucketing.spec --clean --noconfirm

import os

block_cipher = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))
ENTRY = os.path.join(ROOT, "src", "regr_fail_bucketing.py")
SRC = os.path.join(ROOT, "src")

a = Analysis(
    [ENTRY],
    pathex=[SRC],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter", "PIL", "torch"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="regr_fail_bucketing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
