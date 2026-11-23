#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

fetch() {
    local name="$1" url="$2" commit="$3"
    if [ ! -d "third_party/$name/.git" ]; then
        git clone "$url" "third_party/$name"
    fi
    git -C "third_party/$name" checkout --quiet "$commit"
    echo "$name at $(git -C "third_party/$name" rev-parse --short HEAD)"
}

fetch Omni2Sound https://github.com/omni2sound/Omni2Sound.git 0222190690731627f5e98f6e47c43b3535879fcc
fetch soundreactor_vae https://github.com/SonyResearch/soundreactor_vae.git 7d5543aa367c3542d01ba32a10805ba76f57b660
