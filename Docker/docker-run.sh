#!/bin/bash
# ============================================================
# docker-run.sh
# Avvia il container Pinocchio con il codice locale montato.
# Mettilo nella root del progetto Pinocchio.
# ============================================================

PINOCCHIO_DIR="/Users/morgan/LAVORO/ASTRO/PINOCCHIO/DEVEL/Pinocchio"
IMAGE_NAME="pinocchio-dev"
CONTAINER_NAME="pinocchio-workspace"

# Colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Avvio container Pinocchio...${NC}"
echo -e "${YELLOW}Codice montato da: ${PINOCCHIO_DIR}${NC}"
echo -e "${YELLOW}Disponibili: FFTW3, PFFT, heFFTe (CPU), HDF5, OpenMPI${NC}"
echo ""

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --hostname pinocchio-dev \
    -v "${PINOCCHIO_DIR}:/workspace" \
    -v "${PINOCCHIO_DIR}/outputs:/workspace/outputs" \
    --cpus="$(sysctl -n hw.logicalcpu)" \
    --memory="16g" \
    -e "TERM=xterm-256color" \
    -e "PINOCCHIO_SRC=/workspace/src" \
    "${IMAGE_NAME}" bash
