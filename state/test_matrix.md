# Test Matrix — merge/unify-src vs src_Leonardo

Date: 2026-05-27
Branch: merge/unify-src
Binary: OMP=YES build, use_parallel=0 (serial group formation)
Parameter file: example/parameter_file (GridSize=128, RandomSeed=486604)

---

## Reference: src_Leonardo (4 MPI × 1 OMP, heFFTe backend)

```
Total number of peaks:                 107684
Total number of good halos:            88982
Particles with N neighbouring groups:  331379 118001 13583 430 4 0
Total number of accretion events:      268648
Accretion before evaluating merger:    79069
Accretion after evaluating merger:     364
Accretion of filament particles:       48323
Total number of merger events:         18700
Total number of major merger events:   10486
Final number of filament particles:    310917
Final number of particles in halos:    376332
Total number of collapsed particles:   687249
```

---

## merge/unify-src results

### 4 MPI tasks

| OMP threads | good halos | peaks | accretion events | filament particles | PASS? |
|-------------|-----------|-------|-----------------|-------------------|-------|
| 1 | 88982 | 107684 | 268648 | 48323 | ✓ PASS |
| 2 | 88982 | 107684 | 268648 | 48323 | ✓ PASS |
| 4 | 88982 | 107684 | 268648 | 48323 | ✓ PASS |

Detailed (4 MPI × 1 OMP):
```
Total number of peaks:                 107684
Total number of good halos:            88982
Particles with N neighbouring groups:  331379 118001 13583 430 4 0
Total number of accretion events:      268648
Accretion before evaluating merger:    79069
Accretion after evaluating merger:     364
Accretion of filament particles:       48323
Total number of merger events:         18700
Total number of major merger events:   10486
Final number of filament particles:    310917
```

**Verdict: exact match with src_Leonardo reference.**

### 2 MPI tasks

| OMP threads | good halos | peaks | accretion events | filament particles | PASS? |
|-------------|-----------|-------|-----------------|-------------------|-------|
| 1 | 88981 | 107684 | 268658 | 48327 | ✓ PASS* |
| 2 | 88981 | 107684 | 268658 | 48327 | ✓ PASS* |
| 4 | 88981 | 107684 | 268658 | 48327 | ✓ PASS* |

Detailed (2 MPI × 1 OMP):
```
Total number of peaks:                 107684
Total number of good halos:            88981
Particles with N neighbouring groups:  331378 117995 13584 430 4 0
Total number of accretion events:      268658
Accretion of filament particles:       48327
```

*The 2 MPI results differ by 1 halo (88981 vs 88982) from the 4 MPI reference.
This is an MPI boundary effect: different domain decompositions process edge particles
in slightly different sequential order. This is expected and physically correct behavior.
Note: 88981 matches the CLAUDE.md reference exactly (which was from a 2-MPI run).

---

## Key finding

- `use_parallel = 0` in `build_groups.c` enforces the serial group-formation path
- Serial path is fully reproducible: results are **identical across all OMP thread counts** for a given MPI decomposition
- OMP threads benefit the other parallelized steps (FFTs, sort, etc.) without affecting group formation correctness
- 1 MPI task excluded: crashes with "TOO MANY GROUPS" for GridSize=128 (ngroups > Npeaks+2)

---

## Next step

Future parallelization of group formation should follow the src_Leonardo serial algorithm
as reference, adding OpenMP only at a level that preserves the sequential dependency
along the z_c axis (boundary-aware wavefront or task-based approach).
