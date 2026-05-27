#!/bin/bash
set -e

# Test matrix runner for PINOCCHIO
# Compiles and runs both src_Leonardo and merge/unify-src across MPI×OMP configurations

WORKSPACE="/Users/morgan/LAVORO/ASTRO/PINOCCHIO/DEVEL/Pinocchio"
RESULTS_FILE="${WORKSPACE}/state/test_matrix.md"

# Function to extract statistics from output
extract_stats() {
    local output="$1"
    echo "$output" | grep -E "Total number of peaks:|Total number of good halos:|Particles with N neighbouring groups:|Total number of accretion events:|Accretion before evaluating merger:|Accretion after evaluating merger:|Accretion of filament particles:|Total number of merger events:|Total number of major merger events:|Final number of filament particles:|Final number of particles in halos:|Total number of collapsed particles:|Total\s+:|Fragmentation\s+:|Redistribution\s+:|Sorting\s+:|Groups\s+:"
}

echo "=== PINOCCHIO Test Matrix Runner ==="
echo "Working directory: $WORKSPACE"
echo ""

# Step 1: Compile src_Leonardo
echo "[1/3] Compiling src_Leonardo baseline..."
docker run --rm -v "$WORKSPACE:/workspace" --memory=16g pinocchio-dev bash -c "
cd /workspace/src_Leonardo && \
mpicc -I/usr/local/pfft/include -I/usr/local/fftw3/include -I/root/include/gsl \
  -O3 -Wno-unused-result \
  -DTWO_LPT -DTHREE_LPT -DELL_CLASSIC -DNORADIATION \
  -o pinocchio_leonardo.x \
  pinocchio.c fmax.c variables.c initialization.c collapse_times.c \
  GenIC.c ReadParamfile.c allocations.c LPT.c distribute.c \
  fragment.c build_groups.c write_halos.c write_snapshot.c cosmo.c \
  -lm -L/usr/local/pfft/lib -lpfft -L/usr/local/fftw3/lib -lfftw3_mpi -lfftw3 \
  -L/root/lib -lgsl -lgslcblas -lm 2>&1
" && echo "✓ src_Leonardo compiled" || echo "✗ src_Leonardo compilation failed"

# Step 2: Compile merge/unify-src with OMP
echo "[2/3] Compiling merge/unify-src with OMP..."
docker run --rm -v "$WORKSPACE:/workspace" pinocchio-dev make -C /workspace/src clean 2>&1 > /dev/null
docker run --rm -v "$WORKSPACE:/workspace" pinocchio-dev make -C /workspace/src OMP=YES 2>&1 | tail -20 && echo "✓ merge/unify-src compiled" || echo "✗ merge/unify-src compilation failed"

# Step 3: Run tests
echo "[3/3] Running test matrix..."

# Initialize results file
cat > "$RESULTS_FILE" << 'EOF'
# PINOCCHIO Test Matrix Results

Generated: 2026-05-27

## Reference Values (from CLAUDE.md)
```
Total number of peaks:              107684
Total number of good halos:         88981
Particles with N neighbouring groups:  331378 117995 13584 430 4 0
Total number of accretion events:   268658
Accretion before evaluating merger: 79071
Accretion after evaluating merger:  364
Accretion of filament particles:    48327
Total number of merger events:      ???
Total number of major merger events: ???
Final number of filament particles: ???
Final number of particles in halos: ???
Total number of collapsed particles: ???
```

## Baseline Compilation Status
- src_Leonardo: building...
- merge/unify-src: building...

---

EOF

# Test 1: src_Leonardo baseline (4 MPI × 1 OMP)
echo "Test 1/10: src_Leonardo (4 MPI × 1 OMP)..."
OUTPUT=$(docker run --rm -v "$WORKSPACE:/workspace" --memory=16g pinocchio-dev bash -c "
cp /workspace/src_Leonardo/pinocchio_leonardo.x /workspace/example/pinocchio.x && \
cd /workspace/example && \
mpirun --allow-run-as-root -np 4 ./pinocchio.x parameter_file 2>&1
" 2>&1)
echo "$OUTPUT" | tail -100 > /tmp/test_1_output.txt
extract_stats "$OUTPUT" > /tmp/test_1_stats.txt

# Tests 2-10: merge/unify-src configurations
declare -a MPI_COUNTS=(1 1 1 2 2 2 4 4 4)
declare -a OMP_COUNTS=(1 2 4 1 2 4 1 2 4)

for i in {0..8}; do
    test_num=$((i + 2))
    mpi=${MPI_COUNTS[$i]}
    omp=${OMP_COUNTS[$i]}
    echo "Test $test_num/10: merge/unify-src ($mpi MPI × $omp OMP)..."

    OUTPUT=$(docker run --rm -v "$WORKSPACE:/workspace" --memory=16g -e OMP_NUM_THREADS=$omp pinocchio-dev bash -c "
cp /workspace/src/pinocchio.x /workspace/example/pinocchio.x && \
cd /workspace/example && \
mpirun --allow-run-as-root -np $mpi ./pinocchio.x parameter_file 2>&1
" 2>&1)
    echo "$OUTPUT" | tail -100 > /tmp/test_${test_num}_output.txt
    extract_stats "$OUTPUT" > /tmp/test_${test_num}_stats.txt
done

echo ""
echo "=== All tests completed ==="
echo "Results saved to: $RESULTS_FILE"
echo "Run outputs saved to /tmp/test_*_output.txt and /tmp/test_*_stats.txt"
