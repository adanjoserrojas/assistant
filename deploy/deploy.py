"""Build the Lambda deployment package.

    python deploy.py            build build/calendar-agent.zip
    python deploy.py --upload   build, then push it to the live function

Contains no credentials and no account-specific values -- everything that
differs between users is an environment variable.

Two things here are not obvious, and are why this is a script rather than a
line in the README:

1. Wheels must be built for Linux. A plain `pip install --target` on Windows
   pulls win_amd64 wheels for cryptography, which fail to import on Lambda
   with an error that looks like a corrupt package rather than a wrong one.
2. google-api-python-client bundles a discovery document for every Google API
   -- 586 files, 100 MB. This project needs exactly one, and dropping the rest
   takes the package from 135 MB to 27 MB.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "calendar-agent")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Must match the Lambda's configured runtime and architecture.
PYTHON_VERSION = "3.13"
PLATFORM = "manylinux2014_x86_64"

# boto3 ships with the Lambda runtime, so it is deliberately absent here.
DEPENDENCIES = ["google-api-python-client", "google-auth", "tzdata"]

SOURCE_MODULES = [
    "agent.py",
    "calendar_client.py",
    "llm_client.py",
    "scheduler.py",
    "validator.py",
    "models.py",
    "config.py",
]

KEEP_DISCOVERY_DOC = "calendar.v3.json"

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build"
PACKAGE = BUILD / "package"
ZIP_PATH = BUILD / "calendar-agent.zip"


def install_dependencies():
    print(f"installing {len(DEPENDENCIES)} deps for {PLATFORM} / py{PYTHON_VERSION}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--platform", PLATFORM,
            "--only-binary=:all:",
            "--python-version", PYTHON_VERSION,
            "--target", str(PACKAGE),
            *DEPENDENCIES,
        ],
        check=True,
    )


def trim():
    """Drop everything Lambda will never read."""
    documents = PACKAGE / "googleapiclient" / "discovery_cache" / "documents"
    dropped = 0
    if documents.is_dir():
        for doc in documents.iterdir():
            if doc.name != KEEP_DISCOVERY_DOC:
                doc.unlink()
                dropped += 1
    print(f"dropped {dropped} unused discovery documents")

    for cache in PACKAGE.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def directory_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build():
    shutil.rmtree(BUILD, ignore_errors=True)
    PACKAGE.mkdir(parents=True)

    install_dependencies()
    trim()

    for module in SOURCE_MODULES:
        source = ROOT / module
        if not source.exists():
            raise FileNotFoundError(f"missing source module: {module}")
        shutil.copy2(source, PACKAGE / module)
    print(f"copied {len(SOURCE_MODULES)} source modules")

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", str(PACKAGE))

    unzipped = directory_size(PACKAGE) / 1024 / 1024
    zipped = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"\n{ZIP_PATH.name}  {zipped:.1f} MB zipped / {unzipped:.0f} MB unzipped")
    if zipped >= 50:
        print("WARNING: over Lambda's 50 MB direct-upload limit -- upload via S3")
    return ZIP_PATH


def upload(zip_path):
    """Requires lambda:UpdateFunctionCode, which the runtime role does not have."""
    import boto3

    print(f"uploading to {FUNCTION_NAME} in {AWS_REGION}")
    response = boto3.client("lambda", region_name=AWS_REGION).update_function_code(
        FunctionName=FUNCTION_NAME,
        ZipFile=zip_path.read_bytes(),
        Publish=True,
    )
    print(f"done -- version {response['Version']}, {response['CodeSize']} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Lambda deployment package")
    parser.add_argument("--upload", action="store_true", help="push to the live function")
    args = parser.parse_args()

    archive = build()
    if args.upload:
        upload(archive)
    else:
        print("\nnot uploaded -- pass --upload, or upload the zip in the console")
