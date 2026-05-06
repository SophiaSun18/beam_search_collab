#!/usr/bin/env bash
# Install CUDA toolkit + GCC into ~/.local/ without root.
# Targets x86_64 Linux. GCC extracted from Ubuntu 24.04 .deb packages.
#
# Usage:
#   bash install_cuda_gcc.sh              # install both
#   bash install_cuda_gcc.sh --cuda-only  # CUDA only
#   bash install_cuda_gcc.sh --gcc-only   # GCC only
#
# Override install prefix:
#   INSTALL_PREFIX=/opt/local bash install_cuda_gcc.sh

set -euo pipefail

PREFIX="${INSTALL_PREFIX:-$HOME/.local}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# ======================== Versions ========================
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"

GCC_MAJOR=14
GCC_VER="14.2.0-4ubuntu2~24.04.1"
_MAIN="http://security.ubuntu.com/ubuntu/pool/main/g/gcc-${GCC_MAJOR}"
_UNIV="http://security.ubuntu.com/ubuntu/pool/universe/g/gcc-${GCC_MAJOR}"
GCC_URLS=(
    "${_MAIN}/gcc-${GCC_MAJOR}-base_${GCC_VER}_amd64.deb"
    "${_MAIN}/libgcc-${GCC_MAJOR}-dev_${GCC_VER}_amd64.deb"
    "${_UNIV}/cpp-${GCC_MAJOR}_${GCC_VER}_amd64.deb"
    "${_UNIV}/gcc-${GCC_MAJOR}_${GCC_VER}_amd64.deb"
    "${_UNIV}/g++-${GCC_MAJOR}_${GCC_VER}_amd64.deb"
    "${_UNIV}/libstdc++-${GCC_MAJOR}-dev_${GCC_VER}_amd64.deb"
)
# ==========================================================

require() {
    for cmd in "$@"; do
        command -v "$cmd" &>/dev/null || {
            echo "ERROR: missing required command: $cmd" >&2
            exit 1
        }
    done
}

install_cuda() {
    require wget
    echo "==> Downloading CUDA toolkit (~5 GB)..."
    wget -c -q --show-progress -O "${WORKDIR}/cuda.run" "$CUDA_RUNFILE_URL"

    echo "==> Installing CUDA toolkit to ${PREFIX}..."
    bash "${WORKDIR}/cuda.run" \
        --toolkit --silent \
        --toolkitpath="$PREFIX" \
        --override --no-man-page

    echo "==> CUDA toolkit installed."
}

install_gcc() {
    require wget ar tar
    mkdir -p "${WORKDIR}"/{debs,root}

    echo "==> Downloading GCC ${GCC_MAJOR} packages..."
    for url in "${GCC_URLS[@]}"; do
        echo "    ${url##*/}"
        wget -c -q --show-progress -P "${WORKDIR}/debs" "$url"
    done

    echo "==> Extracting packages..."
    for deb in "${WORKDIR}"/debs/*.deb; do
        mkdir -p "${WORKDIR}/tmp"
        cd "${WORKDIR}/tmp"
        ar x "$deb"
        if ls data.tar.* &>/dev/null; then
            tar xf data.tar.* -C "${WORKDIR}/root"
        fi
        cd "${WORKDIR}"
        rm -rf "${WORKDIR}/tmp"
    done

    echo "==> Installing GCC ${GCC_MAJOR} to ${PREFIX}..."
    local src="${WORKDIR}/root/usr"
    [ -d "$src/bin" ]     && cp -a "$src/bin/"*     "${PREFIX}/bin/"
    [ -d "$src/include" ] && cp -a "$src/include/"* "${PREFIX}/include/"
    [ -d "$src/share" ]   && cp -a "$src/share/"*   "${PREFIX}/share/" 2>/dev/null || true
    [ -d "$src/lib" ]     && cp -a "$src/lib/"*     "${PREFIX}/lib/"
    [ -d "$src/lib64" ]   && cp -a "$src/lib64/"*   "${PREFIX}/lib/"

    cd "${PREFIX}/bin"
    ln -sf "gcc-${GCC_MAJOR}" gcc
    ln -sf "gcc-${GCC_MAJOR}" cc
    ln -sf "g++-${GCC_MAJOR}" g++
    ln -sf "g++-${GCC_MAJOR}" c++
    ln -sf "cpp-${GCC_MAJOR}" cpp
    for t in gcc-ar gcc-nm gcc-ranlib gcov gcov-dump gcov-tool; do
        [ -f "${t}-${GCC_MAJOR}" ] && ln -sf "${t}-${GCC_MAJOR}" "$t" || true
    done
    cd - >/dev/null

    echo "==> GCC ${GCC_MAJOR} installed."
}

DO_CUDA=1
DO_GCC=1
for arg in "$@"; do
    case "$arg" in
        --cuda-only) DO_GCC=0 ;;
        --gcc-only)  DO_CUDA=0 ;;
        -h|--help)
            echo "Usage: $0 [--cuda-only | --gcc-only]"
            echo "Set INSTALL_PREFIX to override ~/.local"
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "${PREFIX}"/{bin,lib,share,include}

[ "$DO_CUDA" = 1 ] && install_cuda
[ "$DO_GCC" = 1 ]  && install_gcc

cat <<DONE

========================================
 Installation complete!
========================================

Add to your shell profile:

  export PATH="${PREFIX}/bin:\$PATH"
  export LD_LIBRARY_PATH="${PREFIX}/lib64:${PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

Verify:
  nvcc --version
  gcc --version

DONE
