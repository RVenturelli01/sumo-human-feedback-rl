#!/usr/bin/env bash
#
# Build the environment and fetch the data needed to run the experiments.
#
#     ./setup.sh
#     conda activate sumo-rlhf
#
# This script cannot activate the environment for you: it runs in its own shell,
# and `conda activate` only affects the shell that calls it. Hence the two steps.

set -euo pipefail

ENV_NAME="sumo-rlhf"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# --- 1. conda environment ----------------------------------------------------
command -v conda >/dev/null || {
    echo "conda not found. Install miniconda or miniforge first." >&2
    exit 1
}

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    # An environment with the right name but the wrong contents is worse than no
    # environment: everything imports and the numbers quietly differ.
    # An environment with the right name but the wrong contents is worse than no
    # environment: everything imports and the numbers quietly differ.
    say "Environment '${ENV_NAME}' already exists, checking it against environment.yml"
    # --no-capture-output, or conda run does not pass this script to python at all
    # and reports success without having run anything.
    if conda run -n "${ENV_NAME}" --no-capture-output python - "${REPO}/environment.yml" <<'CHECK'
import re
import sys
from importlib.metadata import version, PackageNotFoundError

spec = open(sys.argv[1]).read()
want_python = re.search(r"^\s*-\s*python=([\d.]+)", spec, re.M).group(1)
want = dict(re.findall(r"^\s*-\s*([A-Za-z0-9_.-]+)==([^\s#]+)", spec, re.M))

bad = []
got_python = ".".join(str(n) for n in sys.version_info[:3])
if got_python != want_python:
    bad.append(f"python {got_python}, expected {want_python}")
for pkg, expected in sorted(want.items()):
    try:
        got = version(pkg)
    except PackageNotFoundError:
        bad.append(f"{pkg} missing")
        continue
    # Compare without the local build tag: the reference runs used torch's +cpu
    # wheel, which macOS does not publish.
    if got.split("+")[0] != expected.split("+")[0]:
        bad.append(f"{pkg} {got}, expected {expected}")

for line in bad:
    print(f"  {line}")
sys.exit(1 if bad else 0)
CHECK
    then
        echo "  matches"
    else
        cat >&2 <<MSG

  The existing '${ENV_NAME}' environment does not match environment.yml.
  Remove it and run this again, or install into a different name:

      conda env remove -n ${ENV_NAME}
      ./setup.sh
MSG
        exit 1
    fi
else
    say "Creating the '${ENV_NAME}' environment (this takes a few minutes)"
    conda env create -f "${REPO}/environment.yml"
fi

RUN="conda run -n ${ENV_NAME} --no-capture-output"

# --- 2. the two packages, in editable mode -----------------------------------
# Without these, nothing imports: the algorithm and the environment live in the
# submodules, and the experiment layer is only the configuration around them.
say "Installing the submodules"
if [ ! -f "${REPO}/human-feedback-rl/pyproject.toml" ]; then
    echo "Submodules are missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi
$RUN pip install -q -e "${REPO}/human-feedback-rl" -e "${REPO}/sumo-rl-ego"

# --- 3. SUMO -----------------------------------------------------------------
# libsumo comes from pip and is what the code imports. A system SUMO is only
# needed if you want the GUI; the version still matters, because the simulator
# decides the rollouts.
say "Checking SUMO"
$RUN python -c "import libsumo; print('  libsumo', libsumo.__version__ if hasattr(libsumo,'__version__') else 'ok')"
command -v sumo >/dev/null \
    && echo "  system sumo: $(sumo --version 2>/dev/null | head -1)" \
    || echo "  system sumo not in PATH (fine unless you want the GUI; SUMO_HOME must be set for it)"

# --- 4. demonstrations -------------------------------------------------------
# The dataset repository on Hugging Face is PRIVATE. Without access the download
# fails with a 401 and there is nothing this script can do about it: ask the
# author to grant access to `Andrea02/sumo-rlhf-datasets`.
say "Fetching the demonstrations"
if ! $RUN python -c "
from huggingface_hub import HfApi
import sys
try:
    HfApi().whoami()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    cat >&2 <<'MSG'
  Not authenticated with Hugging Face.

  The demonstrations live in a private dataset repository, so an anonymous
  download returns 401. Log in first:

      hf auth login

  and make sure your account has access to Andrea02/sumo-rlhf-datasets.
MSG
    exit 1
fi
$RUN python "${REPO}/experiments/download_datasets.py"

# --- 5. checksums ------------------------------------------------------------
# The demonstrations decide the experiment as much as the code does: a different
# expert dataset gives different numbers with no error anywhere.
say "Verifying the datasets"
cd "${REPO}"
if shasum -a 256 -c datasets/SHA256SUMS; then
    echo "  all four datasets match the reference ones"
else
    echo "  CHECKSUM MISMATCH -- the numbers will not reproduce" >&2
    exit 1
fi

say "Done. Now run:  conda activate ${ENV_NAME}"
