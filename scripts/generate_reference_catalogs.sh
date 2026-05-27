#!/bin/bash
# generate_reference_catalogs.sh
# Compila src_Leonardo e genera i cataloghi di riferimento in state/reference_catalogs/
#
# Uso: ./scripts/generate_reference_catalogs.sh [NP]
#   NP = numero di task MPI (default: 4)
#
# Prerequisiti: docker image pinocchio-dev disponibile
# Output: state/reference_catalogs/pinocchio.*.reference.*.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NP="${1:-4}"
REF_DIR="${REPO_ROOT}/state/reference_catalogs"
TMP_DIR="${REF_DIR}/run_tmp"
SRC_LEONARDO="${REPO_ROOT}/src_Leonardo"
PINOCCHIO_DIR="${REPO_ROOT}"
IMAGE="pinocchio-dev"

echo "=== Generazione cataloghi di riferimento ==="
echo "  src: src_Leonardo"
echo "  NP:  ${NP} task MPI"
echo "  out: ${REF_DIR}"
echo ""

# --- 1. Compilazione src_Leonardo ---
echo "[1/3] Compilazione src_Leonardo..."
docker run --rm \
    -v "${PINOCCHIO_DIR}:/workspace" \
    --cpus="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)" \
    --memory="16g" \
    "${IMAGE}" \
    bash -c "cd /workspace/src_Leonardo && make clean && make SYSTYPE=docker GPU=NO OMP=YES 2>&1 | tail -5"
echo "  -> src_Leonardo/PinocchioFinal_GPU.x pronto"

# --- 2. Preparazione directory run ---
echo "[2/3] Preparazione run directory..."
mkdir -p "${TMP_DIR}"
cp "${REPO_ROOT}/example/parameter_file" "${TMP_DIR}/"
cp "${REPO_ROOT}/example/outputs" "${TMP_DIR}/"

# Usa RunFlag "reference" per distinguere i file di output
sed -i.bak 's/RunFlag[[:space:]].*$/RunFlag                reference/' "${TMP_DIR}/parameter_file"
rm -f "${TMP_DIR}/parameter_file.bak"

echo "  RunFlag: reference"
echo "  GridSize, RandomSeed: come in example/parameter_file"

# --- 3. Run ---
echo "[3/3] Esecuzione src_Leonardo (${NP} MPI tasks)..."
docker run --rm \
    -v "${PINOCCHIO_DIR}:/workspace" \
    --cpus="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)" \
    --memory="16g" \
    "${IMAGE}" \
    bash -c "cd /workspace/state/reference_catalogs/run_tmp && \
             mpirun --allow-run-as-root -np ${NP} \
             /workspace/src_Leonardo/PinocchioFinal_GPU.x parameter_file 2>&1 | \
             grep -E 'good halos|peaks|accretion|ERROR|error|WARNING'"

# --- 4. Copia e pulizia ---
echo ""
echo "Cataloghi generati:"
cp "${TMP_DIR}"/*.out "${REF_DIR}/"
ls "${REF_DIR}"/*.out
rm -rf "${TMP_DIR}"

echo ""
echo "=== COMPLETATO ==="
echo "Cataloghi di riferimento salvati in: ${REF_DIR}"
echo ""
echo "Per confrontare un run di test:"
echo "  python scripts/compare_catalogs.py <test_dir>"
