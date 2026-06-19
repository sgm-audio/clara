#!/usr/bin/env bash
set -euo pipefail

# FMA Dataset Downloader
# Downloads fma_metadata.zip + fma_large.zip via aria2 with resume support
#
# Usage:
#   ./download_fma.sh [dest_dir]
#   Default dest_dir: ~/datasets/fma

DEST="${1:-$HOME/datasets/fma}"
mkdir -p "$DEST"

BASE_URL="https://os.unil.cloud.switch.ch/fma"

declare -A FILES=(
  ["fma_metadata.zip"]="f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
  ["fma_large.zip"]="497109f4dd721066b5ce5e5f250ec604dc78939e"
)

aria2c_installed() { command -v aria2c &>/dev/null; }
sha1_installed()   { command -v sha1sum &>/dev/null; }

if ! aria2c_installed; then
  echo "ERROR: aria2c not found. Install it first:"
  echo "  Fedora Atomic: rpm-ostree install aria2 && reboot"
  echo "  Ubuntu/Debian: sudo apt install aria2"
  echo "  macOS:         brew install aria2"
  exit 1
fi

if ! sha1_installed; then
  echo "ERROR: sha1sum not found."
  exit 1
fi

download_file() {
  local filename="$1"
  local url="$BASE_URL/$filename"
  local dest="$DEST/$filename"

  if [[ -f "$dest" ]]; then
    echo "→ $filename already exists at $dest, checking integrity..."
    if verify_checksum "$filename"; then
      echo "  ✓ $filename is complete and valid. Skipping."
      return 0
    else
      echo "  ⚠ $filename is corrupt or incomplete. Re-downloading..."
    fi
  fi

  echo "→ Downloading $filename..."
  echo "  URL: $url"
  echo "  Dest: $dest"
  echo ""

  aria2c \
    --continue=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=10M \
    --max-concurrent-downloads=1 \
    --connect-timeout=30 \
    --timeout=120 \
    --retry-wait=5 \
    --max-tries=5 \
    --human-readable=true \
    --summary-interval=30 \
    --console-log-level=notice \
    --dir="$DEST" \
    --out="$filename" \
    "$url"

  echo ""
}

verify_checksum() {
  local filename="$1"
  local expected="${FILES[$filename]}"
  local dest="$DEST/$filename"

  if [[ ! -f "$dest" ]]; then
    return 1
  fi

  echo "  Verifying SHA1: $(du -h "$dest" | cut -f1)"
  local actual
  actual=$(sha1sum "$dest" | cut -d' ' -f1)

  if [[ "$actual" == "$expected" ]]; then
    echo "  ✓ SHA1: $actual"
    return 0
  else
    echo "  ✗ SHA1 mismatch:"
    echo "    Expected: $expected"
    echo "    Actual:   $actual"
    return 1
  fi
}

echo "============================================"
echo " FMA Dataset Downloader"
echo " Destination: $DEST"
echo "============================================"
echo ""

for filename in "${!FILES[@]}"; do
  download_file "$filename"
done

echo ""
echo "============================================"
echo " All downloads complete. Verifying..."
echo "============================================"
echo ""

all_ok=true
for filename in "${!FILES[@]}"; do
  if verify_checksum "$filename"; then
    :
  else
    all_ok=false
  fi
done

echo ""
if $all_ok; then
  echo "✓ All files verified successfully."
  echo ""
  echo "Next steps:"
  echo "  cd $DEST"
  echo "  unzip fma_metadata.zip"
  echo "  unzip fma_large.zip        # 93GB — will take a while"
else
  echo "✗ Some files failed verification. Re-run the script to retry."
  exit 1
fi
