"""Build the session validator Lambda deployment package.

    python deploy/deploy_validator_handler.py

Drops gym-session-validator.zip in ~/Downloads, ready to upload in the Lambda
console. Nothing is pushed to AWS -- this only builds the artifact.

Separate from deploy_gym_handler.py: a different function, invoked by EventBridge
rather than API Gateway, and it bundles tzdata where that one does not.

Zip layout, flat at the root:

    validate_sesh_handler.py
    config.py
    tzdata/

The handler is flattened out of handlers/ so the function's handler string is
`validate_sesh_handler.lambda_handler`. config.py sits beside it because
`from config import ...` is an absolute import.

tzdata IS bundled here, unlike the gym command zip. validate_sesh_handler builds
LOCAL_TZ = ZoneInfo(TIMEZONE) at module scope, so a missing tzdata is not one
failed request -- it is an init failure that kills every invocation of the
function. 565 KB is cheap next to a cron that silently never runs.
"""

import shutil
from pathlib import Path

# Repo path -> name at the zip root.
SOURCE_MODULES = {
    "handlers/validate_sesh_handler.py": "validate_sesh_handler.py",
    "config.py": "config.py",
}

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build-validator"
PACKAGE = BUILD / "package"
OUTPUT_DIR = Path.home() / "Downloads"
ZIP_PATH = OUTPUT_DIR / "gym-session-validator.zip"

HANDLER_STRING = "validate_sesh_handler.lambda_handler"

REMINDER = f"""
Lambda configuration this zip expects:

    Handler      {HANDLER_STRING}
    Runtime      python3.12 or later
    Timeout      30s is plenty -- it does at most 2 reads and 1 transaction
    Env          TABLE_NAME   (required; config.py raises KeyError without it)
                 TIMEZONE     (optional, defaults to America/New_York)

    IAM          dynamodb:GetItem, Query, PutItem, UpdateItem on the table

Trigger: EventBridge SCHEDULER, not a classic EventBridge rule.

    cron(0 2 * * ? *)  with TimeZone = America/New_York

Classic rules are UTC-only. Hardcoding cron(0 6 * * ? *) works until November,
then EST shifts it to 1am and the run straddles the wrong day boundary.

Smoke test before trusting the schedule -- replays one specific day:

    {{"target_date": "2026-08-19"}}
"""


def copy_sources():
    for source_path, zip_name in SOURCE_MODULES.items():
        source = ROOT / source_path
        if not source.exists():
            raise FileNotFoundError(f"missing source module: {source_path}")
        shutil.copy2(source, PACKAGE / zip_name)
    print(f"copied {len(SOURCE_MODULES)} source modules")


def copy_tzdata():
    """Vendor tzdata into the zip so ZoneInfo works with no layer attached."""
    try:
        import tzdata
    except ImportError as error:
        raise RuntimeError(
            "tzdata is not installed locally, so it cannot be bundled.\n"
            "Run: pip install tzdata"
        ) from error

    source = Path(tzdata.__file__).parent
    shutil.copytree(source, PACKAGE / "tzdata")

    # .pyc from the local interpreter is dead weight and can mismatch the
    # Lambda runtime's Python version.
    for cache in (PACKAGE / "tzdata").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    size = sum(f.stat().st_size for f in (PACKAGE / "tzdata").rglob("*") if f.is_file())
    print(f"bundled tzdata {size / 1024:.0f} KB")


def build():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"no such directory: {OUTPUT_DIR}")

    shutil.rmtree(BUILD, ignore_errors=True)
    PACKAGE.mkdir(parents=True)

    copy_sources()
    copy_tzdata()

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", str(PACKAGE))
    shutil.rmtree(BUILD, ignore_errors=True)

    zipped = ZIP_PATH.stat().st_size / 1024
    print(f"\n{ZIP_PATH}  {zipped:.1f} KB zipped")
    return ZIP_PATH


if __name__ == "__main__":
    build()
    print(REMINDER)
    print("not uploaded -- upload the zip in the Lambda console")
