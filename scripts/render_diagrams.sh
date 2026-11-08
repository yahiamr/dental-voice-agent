#!/usr/bin/env bash
# Renders the README diagrams. Stage: 4.
#
# What this file is:
#   A script that turns each Mermaid source in assets/diagrams/*.mmd into an SVG beside it.
#
# How to understand it:
#   The README shows the SVGs, so GitHub needs no Mermaid support and the look is fixed.
#   mermaid-cli runs through npx (Node is needed; the first run downloads it from npm) and
#   draws with a headless Chrome. mermaid-config.json holds the shared colours and fonts.
#   Edit a .mmd file, run this script, and commit both files.
#
# Usage:
#   scripts/render_diagrams.sh            render every diagram
#   scripts/render_diagrams.sh turn       render assets/diagrams/turn.mmd only
set -euo pipefail

cd "$(dirname "$0")/../assets/diagrams"
MMDC_VERSION="11.4.2"
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=()
  for file in *.mmd; do names+=("${file%.mmd}"); done
fi

for name in "${names[@]}"; do
  echo "Rendering $name.svg"
  npx -y -p "@mermaid-js/mermaid-cli@${MMDC_VERSION}" mmdc \
    --input "$name.mmd" --output "$name.svg" \
    --configFile mermaid-config.json --backgroundColor white --quiet
done
