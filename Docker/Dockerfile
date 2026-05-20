# ============================================================
# Pinocchio Development Container
# Backends: FFTW3 (MPI), PFFT, heFFTe (CPU/FFTW backend)
# MPI: OpenMPI | Compilatore librerie: GCC | Pinocchio: clang
# Architettura: aarch64 (Apple Silicon) e x86_64
# ============================================================
# Build:
#   docker build -t pinocchio-dev .
#
# Run (monta il codice locale):
#   docker run -it --rm \
#     -v /Users/morgan/LAVORO/ASTRO/PINOCCHIO/DEVEL/Pinocchio:/workspace \
#     pinocchio-dev bash
# ============================================================

# --- Stage 1: builder ---
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# --- Dipendenze di sistema ---
RUN apt-get update && apt-get install -y \
    # Build tools — GCC per le librerie (OpenMP, Fortran, aarch64 compat)
    build-essential \
    gcc \
    g++ \
    gfortran \
    # clang per Pinocchio (installato nell'immagine finale, non serve qui)
    cmake \
    ninja-build \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # Networking
    git \
    wget \
    curl \
    # OpenMPI
    openmpi-bin \
    libopenmpi-dev \
    # HDF5 parallel (da apt per semplicita')
    libhdf5-mpi-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Compilatori per le librerie: GCC + MPI wrapper ---
# GCC ha supporto OpenMP nativo su aarch64, clang no senza libomp extra.
# Le librerie (FFTW, PFFT, heFFTe) si compilano con GCC.
# Pinocchio stesso usa clang (configurato nel Makefile, non qui).
ENV CC=mpicc
ENV CXX=mpicxx
ENV FC=mpif90
ENV OMPI_CC=gcc
ENV OMPI_CXX=g++
ENV OMPI_FC=gfortran

# --- FFTW3 da sorgente con MPI + OpenMP ---
# Necessario per PFFT (apt non fornisce FFTW3-MPI con OpenMP insieme).
# FFTW >= 3.3.5 non richiede patch per PFFT.
ENV FFTW_VERSION=3.3.10
ENV FFTW_PREFIX=/usr/local/fftw3

RUN wget -q https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz && \
    tar xzf fftw-${FFTW_VERSION}.tar.gz && \
    cd fftw-${FFTW_VERSION} && \
    # Double precision
    ./configure \
        --prefix=${FFTW_PREFIX} \
        --enable-mpi \
        --enable-openmp \
        --enable-shared \
        --enable-static \
        CC=mpicc \
        CFLAGS="-O3 -ffast-math" && \
    make -j$(nproc) && \
    make install && \
    # Single precision
    ./configure \
        --prefix=${FFTW_PREFIX} \
        --enable-mpi \
        --enable-openmp \
        --enable-shared \
        --enable-static \
        --enable-float \
        CC=mpicc \
        CFLAGS="-O3 -ffast-math" && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf fftw-${FFTW_VERSION} fftw-${FFTW_VERSION}.tar.gz

# --- PFFT ---
# Backend FFT legacy di Pinocchio, usato come riferimento di validazione.
ENV PFFT_PREFIX=/usr/local/pfft

RUN git clone --depth=1 https://github.com/mpip/pfft.git /tmp/pfft && \
    cd /tmp/pfft && \
    ./bootstrap.sh && \
    ./configure \
        --prefix=${PFFT_PREFIX} \
        --with-fftw3=${FFTW_PREFIX} \
        --enable-shared \
        --enable-static \
        --disable-fortran \
        CC=mpicc \
        CFLAGS="-O3" \
        CPPFLAGS="-I${FFTW_PREFIX}/include" \
        LDFLAGS="-L${FFTW_PREFIX}/lib" && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/pfft

# --- heFFTe (CPU backend: FFTW) ---
# Backend GPU-portable. In locale usa FFTW, su Leonardo usa cuFFT.
# Compilato con GCC per coerenza con FFTW/PFFT.
ENV HEFFTE_VERSION=2.4.0
ENV HEFFTE_PREFIX=/usr/local/heffte

RUN wget -q https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz \
        -O heffte-${HEFFTE_VERSION}.tar.gz && \
    tar xzf heffte-${HEFFTE_VERSION}.tar.gz && \
    cmake -S heffte-${HEFFTE_VERSION} -B heffte-build \
        -DCMAKE_INSTALL_PREFIX=${HEFFTE_PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DHeffte_ENABLE_FFTW=ON \
        -DFFTW_ROOT=${FFTW_PREFIX} \
        -DHeffte_ENABLE_CUDA=OFF \
        -DHeffte_ENABLE_ROCM=OFF \
        -DHeffte_ENABLE_ONEAPI=OFF \
        -DHeffte_ENABLE_MPI=ON \
        -DBUILD_SHARED_LIBS=ON && \
    cmake --build heffte-build -j$(nproc) && \
    cmake --install heffte-build && \
    rm -rf heffte-${HEFFTE_VERSION} heffte-${HEFFTE_VERSION}.tar.gz heffte-build

# ============================================================
# --- Stage 2: immagine finale ---
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # GCC runtime (necessario per le librerie compilate con gcc)
    gcc \
    g++ \
    gfortran \
    libgomp1 \
    # clang per compilare Pinocchio
    clang \
    libomp-dev \
    # MPI
    openmpi-bin \
    libopenmpi-dev \
    # HDF5 parallel
    libhdf5-mpi-dev \
    # Tools
    make \
    cmake \
    git \
    vim \
    less \
    wget \
    gdb \
    valgrind \
    # Python per script di validazione
    python3 \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-scipy \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia librerie dallo stage builder
COPY --from=builder /usr/local/fftw3  /usr/local/fftw3
COPY --from=builder /usr/local/pfft   /usr/local/pfft
COPY --from=builder /usr/local/heffte /usr/local/heffte

# --- Variabili d'ambiente ---
# Per compilare le librerie: GCC via MPI wrapper
ENV OMPI_CC=gcc
ENV OMPI_CXX=g++
ENV OMPI_FC=gfortran

# Per compilare Pinocchio: clang via MPI wrapper
# (sovrascrivibile nel Makefile con: CC=mpicc OMPI_CC=clang make)
ENV PINOCCHIO_CC=clang

ENV FFTW_PREFIX=/usr/local/fftw3
ENV PFFT_PREFIX=/usr/local/pfft
ENV HEFFTE_PREFIX=/usr/local/heffte
ENV HDF5_PREFIX=/usr

# PATH (senza usare LD_LIBRARY_PATH indefinita)
ENV PATH="/usr/local/fftw3/bin:/usr/local/pfft/bin:/usr/local/heffte/bin:${PATH}"

# LD_LIBRARY_PATH definita in modo esplicito senza riferimento a se stessa
ENV LD_LIBRARY_PATH="/usr/local/fftw3/lib:/usr/local/pfft/lib:/usr/local/heffte/lib"

ENV PKG_CONFIG_PATH="/usr/local/fftw3/lib/pkgconfig:/usr/local/pfft/lib/pkgconfig:/usr/local/heffte/lib/pkgconfig"

# --- Verifica installazione ---
RUN echo "=== Verifica librerie ===" && \
    ls /usr/local/fftw3/lib/libfftw3_mpi.a  && echo "FFTW3 MPI:    OK" && \
    ls /usr/local/fftw3/lib/libfftw3.a      && echo "FFTW3:        OK" && \
    ls /usr/local/pfft/lib/libpfft.a        && echo "PFFT:         OK" && \
    ls /usr/local/heffte/lib/libheffte*     && echo "heFFTe:       OK" && \
    mpicc --version                          && echo "MPI (gcc):    OK" && \
    clang --version                          && echo "clang:        OK"

WORKDIR /workspace
CMD ["/bin/bash"]
