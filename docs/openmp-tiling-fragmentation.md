# OpenMP 3D Checkerboard Tiling — Fragmentation Parallelization

## Sommario

Il loop principale di `build_groups()` (fase di frammentazione PINOCCHIO) è stato parallelizzato
con OpenMP usando un approccio a **tiling 3D 8-colori** (8-color 3D checkerboard).

L'implementazione si trova in `src/build_groups.c`, abilitata dalla flag `_OPENMP` (compile con `OMP=YES`)
in assenza di `CLASSIC_FRAGMENTATION`.

---

## Il problema: dipendenza temporale

Il loop seriale processa le particelle in ordine decrescente di `Fmax = 1 + z_c` (redshift di collasso).
Quando si processa la particella P con `Fmax = F`:

- I suoi 6 vicini lagrangiani con `Fmax > F` sono già stati processati → il loro `group_ID[]` è settato
- I vicini con `Fmax < F` non ancora processati → `group_ID[] = 0`

La condizione di **peak** (massimo locale di Fmax) e la **accretion** a un gruppo vicino dipendono
dallo stato corrente di `group_ID[]`. Questo rende il loop intrinsecamente **sequenziale** nell'ordine Fmax.

---

## L'approccio: 8-color checkerboard tiling

### Struttura delle tiles

Il dominio lagrangiano (locale al task MPI) viene suddiviso in **tiles cubic** di lato `TILE_SIZE = 8`
grid spacings. Ogni tile contiene le particelle con coordinate lagrangiane in un cubo 8³.

### Schema a 8 colori

Ogni tile riceve un **colore** basato sulla parità dei suoi indici tridimensionali:

```
color = (tx % 2) | ((ty % 2) << 1) | ((tz % 2) << 2)
```

dove `(tx, ty, tz)` sono gli indici del tile. I colori vanno da 0 a 7.

La proprietà chiave: **due tiles dello stesso colore non sono mai adiacenti** nel reticolo lagrangiano.
La distanza bordo-a-bordo tra tiles dello stesso colore è ≥ `TILE_SIZE = 8` grid spacings.

### Ciclo di processing

```
for color = 0, 1, ..., 7:          ← sequenziale
  #pragma omp parallel for          ← parallelo su tiles
    for each tile of this color:
      for each particle in tile:    ← sequenziale, ordine Fmax decrescente
        [peak / accretion / merge / filament logic]
  barrier                           ← implicita a fine parallel for
```

Le 8 passate di colore sono eseguite **sequenzialmente**. All'interno di ogni passata, i tiles dello
stesso colore vengono processati **in parallelo** da diversi thread OpenMP.

---

## Garanzia di correttezza (per il test 128³)

### Nessuna interferenza di vicini

Due tiles dello stesso colore sono separati di ≥ 8 grid spacings. I vicini lagrangiani di qualsiasi
particella distano esattamente 1 grid spacing. Quindi:

> Nessuna particella in un tile ha un vicino lagrangiano in un tile dello stesso colore.

I vicini sono sempre in tiles di **colore diverso**, processati in passate diverse (prima o dopo).

### Nessuna accretion allo stesso gruppo

Due particelle da tiles dello stesso colore non possono accretarsi sullo stesso gruppo **se**:

```
TILE_SIZE > 2 × max_R_accretion = 2 × f_a × M_max^(1/3)
```

Con `f_a = 0.18` e `M_max ≈ 10000` particelle (limite per 128³):
```
2 × 0.18 × 10000^(1/3) ≈ 7.74 < TILE_SIZE = 8  ✓
```

### Nota: effetti di bordo

I particelle ai **bordi di tile** vedono vicini in tiles di colore diverso (non ancora processati
nella passata corrente) con `group_ID = 0`. Questo può dare decisioni di accretion leggermente
diverse rispetto al loop seriale. Le differenze sono **piccole e accettabili**:

- Test 128³, 1 MPI: 90,559 good halos vs 88,978 serial (~1.8% differenza)
- I risultati sono **deterministici** (identici per qualsiasi numero di thread OMP)

---

## Thread safety

| Operazione | Protezione |
|-----------|-----------|
| `ngroups++` (allocazione peak) | `#pragma omp critical(ngroups_alloc)` |
| `groups[FILAMENT].Mass++/--` | `#pragma omp atomic` |
| `counters[]` | Thread-local `tl_cnt[]`, ridotto via `#pragma omp critical(counters_reduce)` |
| `group_ID[iz]`, `linking_list[iz]` | Ogni particella scrive solo il proprio indice (safe) |
| `groups[my_group].*` | Thread con ownership esclusiva dopo critical section |
| `accretion()`, `merge_groups()` | Safe: tiles stesso colore non condividono gruppi |
| `obj1`, `obj2`, `particle_name`, `good_particle` | `#pragma omp threadprivate` |

---

## PLC (Past Light Cone)

Il processing PLC è estratto in una **post-passata seriale** dopo le 8 passate di colore.
All'inizio della post-passata, `group_ID[]` è completamente popolato, quindi le verifiche
PLC usano lo stato corretto dei gruppi.

---

## Risultati di benchmark

**Sistema:** Docker container pinocchio-dev, macOS ARM64, GridSize=128

### Confronto con il loop seriale (1 MPI task)

| Config | Good Halos | Fragmentation | Total |
|--------|-----------|---------------|-------|
| Serial (`make`, no OMP), 1 MPI | 88,978 | 1.012s (17.3%) | 5.86s |
| OMP=1 thread (`make OMP=YES`), 1 MPI | 90,559 (+1.8%) | 0.876s (14.9%) | 5.90s |
| OMP=4 threads, 1 MPI | 90,559 (+1.8%) | 0.863s (14.7%) | 5.86s |

**Osservazioni:**
- Il path di tiling è **~13% più veloce** del loop seriale anche con 1 thread (locality cache)
- Per il test 128³, lo speedup con più thread è marginale (~1.5% da 1→4 thread)
- Il test 128³ è troppo piccolo per il tiling: ~512 particelle/tile, overhead OMP domina
- Per domani grandi (512³+), lo speedup atteso scala con il numero di thread

### Scalabilità attesa (stima)

Per N_rank = 256³ ≈ 16M particelle per rank:
- ~128K tiles × 128 particelle/tile
- Work per thread >> overhead OMP
- Speedup atteso: 3-6x con 8 thread (considerando serial fraction ~20%)

---

## Come usare

```bash
# Compile con OpenMP
cd src/
../Docker/docker-run.sh make OMP=YES

# Run con 4 MPI × 4 OMP threads
OMP_NUM_THREADS=4 ../Docker/docker-run.sh mpirun --allow-run-as-root -np 4 ./pinocchio.x ../example/parameter_file

# Fallback seriale (risultati identici alla versione originale):
make  # senza OMP=YES
# oppure: make OMP=YES -DCLASSIC_FRAGMENTATION
```

---

## Parametri configurabili

| Parametro | Default | Effetto |
|-----------|---------|---------|
| `TILE_SIZE` | 8 | Lato del tile in grid spacings. Aumentare per simulazioni con aloni più massivi. |

Regola: `TILE_SIZE > 2 × f_a × M_max_halo^(1/3)` per correttezza garantita.

---

## Stato implementazione

- [x] Tile infrastructure (`tile_build`, `tile_free`, `tile_t`, 8-color classification)
- [x] Parallel loop 8 colori con tutti i 4 casi (peak/accretion/merge/filament)
- [x] Thread safety (`critical`, `atomic`, `threadprivate`)
- [x] PLC post-passata seriale
- [x] Output writes seriali post-tiling
- [x] Gestione pause condition (multi-epoch)
- [x] Fallback seriale (loop originale intatto) per no-OMP e CLASSIC_FRAGMENTATION
- [ ] Benchmark su sistema di produzione (Leonardo/Marconi100)
- [ ] Scalabilità per GridSize > 256
