#!/usr/bin/env bash
# Fetches the MD17/MD22 benchmark files into data/bronze/{md17,md22}/.
#
# These are public datasets (sGDML / quantum-machine.org) - nobody needs a
# manual copy of the ~1GB payload, everyone just runs this script. See
# glossary.md section 3 for what's expected to land where, and section 5
# for why theory level (DFT vs CCSD vs CCSD(T)) matters once loaded.
#
# Usage: scripts/download_bronze_data.sh [--force]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRONZE_DIR="$ROOT_DIR/data/bronze"
FORCE="${1:-}"

QM_BASE="http://quantum-machine.org/gdml/data/npz"
SGDML_BASE="https://sgdml.org/secure_proxy.php?file=repo/datasets"

# One canonical source file per molecule+theory level - matches
# glossary.md's inventory. Where both an .npz and a .zip existed for the
# same data, only the .npz is fetched (the .zip was documented as
# redundant, see glossary.md section 3).
#
# Format: "dataset/filename|source_url"
FILES=(
  "md17/md17_ethanol.npz|$QM_BASE/md17_ethanol.npz"
  "md17/ethanol_ccsd_t.zip|$QM_BASE/ethanol_ccsd_t.zip"
  "md17/benzene2018_dft.npz|$QM_BASE/benzene2018_dft.npz"
  "md17/azobenzene_dft.npz|$QM_BASE/azobenzene_dft.npz"
  "md17/paracetamol_dft.npz|$QM_BASE/paracetamol_dft.npz"
  "md17/aspirin_ccsd.zip|$QM_BASE/aspirin_ccsd.zip"
  "md17/malonaldehyde_ccsd_t.zip|$QM_BASE/malonaldehyde_ccsd_t.zip"
  "md17/toluene_ccsd_t.zip|$QM_BASE/toluene_ccsd_t.zip"
  "md17/md17_uracil.npz|$QM_BASE/md17_uracil.npz"
  "md22/md22_AT-AT-CG-CG.npz|$QM_BASE/md22_AT-AT-CG-CG.npz"
  "md22/md22_DHA.npz|$QM_BASE/md22_DHA.npz"
  "md22/md22_buckyball-catcher.npz|$QM_BASE/md22_buckyball-catcher.npz"
  "md22/md22_double-walled_nanotube.npz|$SGDML_BASE/md22_double-walled_nanotube.npz"
)

mkdir -p "$BRONZE_DIR/md17" "$BRONZE_DIR/md22"

# A file that "exists" isn't necessarily real data - if this repo was
# cloned without git-lfs installed, these exact paths get populated with
# ~130-byte LFS pointer text files instead of the real content. Treat
# anything implausibly small, or literally an LFS pointer, as missing.
is_real_file() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local size
  size=$(wc -c < "$path" | tr -d ' ')
  [[ "$size" -gt 1024 ]] || return 1
  head -c 200 "$path" | grep -q "git-lfs.github.com/spec" && return 1
  return 0
}

for entry in "${FILES[@]}"; do
  target="${entry%%|*}"
  url="${entry##*|}"
  dest="$BRONZE_DIR/$target"

  if is_real_file "$dest" && [[ "$FORCE" != "--force" ]]; then
    echo "skip (already have): $target"
    continue
  fi

  echo "downloading: $target"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
done

echo ""
echo "done. data/bronze/ now contains:"
du -sh "$BRONZE_DIR"/md17 "$BRONZE_DIR"/md22
