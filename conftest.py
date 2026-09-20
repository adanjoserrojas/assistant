"""Both source roots on the path, for every test, from any directory.

The repo is deliberately flat-importable: the Lambda zips put every module at
their root, so `import config` and `from scheduler import ...` have to resolve
the same way here. calendar-agent/ is a source directory, not a package -- the
hyphen makes it unimportable as one -- so it goes on sys.path instead.

pytest loads this from the rootdir before collection, which is early enough for
the module-scope `import config` in every test file.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

for path in (ROOT, os.path.join(ROOT, "calendar-agent")):
    if path not in sys.path:
        sys.path.insert(0, path)
