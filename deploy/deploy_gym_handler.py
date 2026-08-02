"""Build the gym command Lambda deployment package.

    python deploy_gym_handler.py

Drops gym-command-handler.zip in ~/Downloads, ready to upload in the Lambda
console. Nothing is pushed to AWS -- this only builds the artifact.

This is separate from deploy.py, which packages the calendar agent: a different
function, a different file list, and dependencies this one does not need.

Three things here are not obvious:

1. Layout is fixed by the configured handler string,
   `handlers.gym_command_handler.lambda_handler` (planning.md step 3). The
   handler must live under `handlers/`, and config.py must sit at the zip root
   because `from config import ...` is an absolute import. Flattening the zip
   breaks the handler string; nesting config.py breaks the import.
2. tzdata must be bundled. The handler calls `ZoneInfo(TIMEZONE)` in
   `current_timestamps()`, and the Lambda runtime image carries no system
   timezone database -- without the package, every command that stamps a time
   dies with ZoneInfoNotFoundError. deploy.py bundles it for the same reason.
3. Uploading replaces *all* function code, so tzdata has to be in every zip,
   not just the first one.

boto3 ships with the Lambda runtime and is deliberately not bundled.
"""

import shutil
import subprocess
import sys
from pathlib import Path

DEPENDENCIES = ["tzdata"]

# Paths are relative to the zip root and mirror the repo layout.
SOURCE_MODULES = [
    "config.py",
    "handlers/gym_command_handler.py",
]

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build-gym"
PACKAGE = BUILD / "package"
OUTPUT_DIR = Path.home() / "Downloads"
ZIP_PATH = OUTPUT_DIR / "gym-command-handler.zip"


def install_dependencies():
    """tzdata is a pure-Python wheel, so it needs no platform pinning."""
    print(f"installing {len(DEPENDENCIES)} dep(s)")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--target", str(PACKAGE),
            *DEPENDENCIES,
        ],
        check=True,
    )


def copy_sources():
    for module in SOURCE_MODULES:
        source = ROOT / module
        if not source.exists():
            raise FileNotFoundError(f"missing source module: {module}")
        destination = PACKAGE / module
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"copied {len(SOURCE_MODULES)} source modules")


def trim():
    for cache in PACKAGE.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def build():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"no such directory: {OUTPUT_DIR}")

    shutil.rmtree(BUILD, ignore_errors=True)
    PACKAGE.mkdir(parents=True)

    install_dependencies()
    copy_sources()
    trim()

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", str(PACKAGE))
    shutil.rmtree(BUILD, ignore_errors=True)

    zipped = ZIP_PATH.stat().st_size / 1024
    print(f"\n{ZIP_PATH}  {zipped:.0f} KB zipped")
    return ZIP_PATH


if __name__ == "__main__":
    build()
    print("\nnot uploaded -- upload the zip in the Lambda console")
