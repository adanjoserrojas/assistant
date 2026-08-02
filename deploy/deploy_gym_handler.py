"""Build the gym command Lambda deployment package.

    python deploy/deploy_gym_handler.py

Drops gym-command-handler.zip in ~/Downloads, ready to upload in the Lambda
console. Nothing is pushed to AWS -- this only builds the artifact.

This is separate from deploy.py, which packages the calendar agent: a different
function, a different file list, and dependencies this one does not need.

Two files, both flat at the zip root:

    gym_command_handler.py
    config.py

The handler is flattened out of handlers/ to match the function's configured
handler string, `gym_command_handler.lambda_handler`. config.py sits beside it
because `from config import ...` is an absolute import. Note this differs from
planning.md step 3, which documents the nested `handlers.` form -- the live
function is the authority.

Nothing else is bundled. boto3 ships with the Lambda runtime; tzdata is
expected to already be present on the function (see WARNING below).
"""

import shutil
from pathlib import Path

# Repo path -> name at the zip root.
SOURCE_MODULES = {
    "handlers/gym_command_handler.py": "gym_command_handler.py",
    "config.py": "config.py",
}

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build-gym"
PACKAGE = BUILD / "package"
OUTPUT_DIR = Path.home() / "Downloads"
ZIP_PATH = OUTPUT_DIR / "gym-command-handler.zip"

WARNING = """
WARNING: this zip carries no tzdata, and uploading replaces all function code.
The handler calls ZoneInfo(TIMEZONE); if the function has no tzdata layer,
every command that stamps a time fails with ZoneInfoNotFoundError.
"""


def copy_sources():
    for source_path, zip_name in SOURCE_MODULES.items():
        source = ROOT / source_path
        if not source.exists():
            raise FileNotFoundError(f"missing source module: {source_path}")
        shutil.copy2(source, PACKAGE / zip_name)
    print(f"copied {len(SOURCE_MODULES)} source modules")


def build():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"no such directory: {OUTPUT_DIR}")

    shutil.rmtree(BUILD, ignore_errors=True)
    PACKAGE.mkdir(parents=True)

    copy_sources()

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", str(PACKAGE))
    shutil.rmtree(BUILD, ignore_errors=True)

    zipped = ZIP_PATH.stat().st_size / 1024
    print(f"\n{ZIP_PATH}  {zipped:.1f} KB zipped")
    return ZIP_PATH


if __name__ == "__main__":
    build()
    print(WARNING)
    print("not uploaded -- upload the zip in the Lambda console")
