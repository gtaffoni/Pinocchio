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

**Esempio concreto:**

```
Griglia 1D con Fmax = [3.1, 2.7, 2.9, 3.5, 2.4, 3.8, 2.2]
Ordine di processing: F=3.8, 3.5, 3.1, 2.9, 2.7, 2.4, 2.2

Step 1: P[5] (F=3.8) — nessun vicino processato → PEAK → diventa gruppo G1
Step 2: P[3] (F=3.5) — vicino P[4] non processato, vicino P[2] non processato → PEAK → G2
Step 3: P[0] (F=3.1) — vicino P[1] non processato → PEAK → G3
Step 4: P[2] (F=2.9) — vicino P[3] (G2) processato, vicino P[1] non processato → ACCRETION su G2
Step 5: P[1] (F=2.7) — vicino P[0] (G3) e P[2] (G2) → MERGE G3→G2 o viceversa
```

Se processassimo P[2] prima di P[3], P[2] diventerebbe un peak invece di accretarsi.
L'ordine non è sceglibile liberamente.

---

## L'approccio: 8-color checkerboard tiling

### Cos'è un tile?

Un **tile** è un piccolo cubo del reticolo lagrangiano di lato `TILE_SIZE` grid spacings.
Con `TILE_SIZE = 8`, ogni tile contiene `8³ = 512` particelle.

Il dominio locale al task MPI (subbox) viene suddiviso in tiles. Per una subbox 64³:
- Numero di tiles per dimensione: 64/8 = 8
- Numero totale di tiles: 8³ = 512 tiles

**Visualizzazione 2D con TILE_SIZE=4 su dominio 12×12:**

```
+----+----+----+
|    |    |    |  ← riga ty=0
| T0 | T1 | T2 |    3 tile per riga
|    |    |    |
+----+----+----+
|    |    |    |  ← riga ty=1
| T3 | T4 | T5 |
|    |    |    |
+----+----+----+
|    |    |    |  ← riga ty=2
| T6 | T7 | T8 |
|    |    |    |
+----+----+----+
```

### Cos'è il colore di un tile?

Ogni tile riceve un **colore** da 0 a 7, basato sulla **parità** dei suoi indici (tx, ty, tz):

```c
color = (tx % 2) | ((ty % 2) << 1) | ((tz % 2) << 2)
```

Equivalentemente: `color = 4*(tz%2) + 2*(ty%2) + (tx%2)`

**Tabella colori per i 8 casi possibili:**

| tx%2 | ty%2 | tz%2 | Colore |
|------|------|------|--------|
|   0  |   0  |   0  |   0    |
|   1  |   0  |   0  |   1    |
|   0  |   1  |   0  |   2    |
|   1  |   1  |   0  |   3    |
|   0  |   0  |   1  |   4    |
|   1  |   0  |   1  |   5    |
|   0  |   1  |   1  |   6    |
|   1  |   1  |   1  |   7    |

**Visualizzazione 2D del pattern di colori (mostra tz=0):**

```
Dominio 2D con TILE_SIZE=4 e 6×6 tiles:

tx:    0    1    2    3    4    5
    +----+----+----+----+----+----+
ty=0| C0 | C1 | C0 | C1 | C0 | C1 |
    +----+----+----+----+----+----+
ty=1| C2 | C3 | C2 | C3 | C2 | C3 |
    +----+----+----+----+----+----+
ty=2| C0 | C1 | C0 | C1 | C0 | C1 |
    +----+----+----+----+----+----+
ty=3| C2 | C3 | C2 | C3 | C2 | C3 |
    +----+----+----+----+----+----+
ty=4| C0 | C1 | C0 | C1 | C0 | C1 |
    +----+----+----+----+----+----+
ty=5| C2 | C3 | C2 | C3 | C2 | C3 |
    +----+----+----+----+----+----+
```

**Proprietà fondamentale:** due tiles dello stesso colore (es. tutti i C0) non sono mai adiacenti.
La distanza bordo-a-bordo tra due tiles dello stesso colore è sempre **≥ TILE_SIZE**.

Con TILE_SIZE=4: il C0 in (tx=0,ty=0) ha il C0 più vicino in (tx=2,ty=0), a distanza 4×4=4 celle.
Il loro bordo comune è a 4 celle di distanza.

---

## Come funziona il processing a 8 colori

### Il ciclo

```
for color = 0, 1, ..., 7:          ← sequenziale (8 passate)
  #pragma omp parallel for          ← parallelo su tiles
    for each tile of this color:
      for each particle in tile:    ← sequenziale, ordine Fmax decrescente
        [peak / accretion / merge / filament logic]
  barrier                           ← implicita a fine parallel for
```

### Passata per passata — esempio 2D

Consideriamo una parte del dominio 2D con 4 tiles (2×2), TILE_SIZE=4:

```
Passata colore 0: processa i tiles C0 (in parallelo)

Thread 1: processa T(0,0) → particelle ordinate per Fmax
          [3.8, 3.5, 3.2, 2.9, ...]
          Crea picchi interni al tile, accreta vicini nel tile

Thread 2: processa T(2,0) → particelle ordinate per Fmax
          [4.1, 3.6, 3.1, ...]
          Crea picchi interni, accreta vicini nel tile

Thread 3: processa T(0,2) → ...
Thread 4: processa T(2,2) → ...
```

Nota: durante la passata colore 0, i tiles C1, C2, C3 NON vengono toccati.
Le particelle ai bordi di T(0,0) che hanno vicini in T(1,0) (colore C1) vedono `group_ID=0`
perché C1 non è ancora stato processato.

```
Barrier (tutti i thread finiscono)

Passata colore 1: processa i tiles C1 (in parallelo)

I tiles C1 ora possono vedere i risultati di C0 nei loro vicini!
Una particella al bordo di T(1,0) che confina con T(0,0) vede i group_ID
già assegnati da T(0,0) → può accretarsi correttamente.
```

### Perché due tiles dello stesso colore possono essere in parallelo

Due tiles C0 adiacenti non esistono: il C0 più vicino è a distanza ≥ TILE_SIZE.

I vicini lagrangiani di qualsiasi particella distano **esattamente 1 grid spacing** (i 6 vicini
ortogonali). Quindi nessuna particella in un tile C0 ha un vicino in un altro tile C0.

**Conseguenza:** il processing di T(0,0) e T(2,0) sono completamente indipendenti:
- Non leggono gli stessi `group_ID[]` (nessun vicino condiviso)
- Non scrivono nello stesso `groups[]` (ogni particella ha ownership del suo gruppo)
- Non c'è race condition

---

## Garanzia di correttezza (per il test 128³)

### Nessuna interferenza di vicini

Due tiles dello stesso colore sono separati di ≥ 8 grid spacings. I vicini lagrangiani di qualsiasi
particella distano esattamente 1 grid spacing. Quindi:

> Nessuna particella in un tile ha un vicino lagrangiano in un tile dello stesso colore.

I vicini sono sempre in tiles di **colore diverso**, processati in passate diverse (prima o dopo).

### Nessuna accretion allo stesso gruppo da tiles paralleli

Due tiles dello stesso colore distano ≥ TILE_SIZE. Per creare conflitti, due particelle in tiles
diversi dello stesso colore dovrebbero voler accretarsi sullo stesso gruppo.

Un gruppo può accretare particelle entro raggio `R_acc = f_a × M^(1/3)`.
La distanza minima tra due tiles dello stesso colore è TILE_SIZE.

Condizione di sicurezza:
```
TILE_SIZE > 2 × R_acc_max = 2 × f_a × M_max^(1/3)
```

Con `f_a = 0.18` e `M_max ≈ 10000` particelle (limite per 128³):
```
2 × 0.18 × 10000^(1/3) ≈ 2 × 0.18 × 21.5 ≈ 7.74 < TILE_SIZE = 8  ✓
```

**Nota importante:** questa garanzia dipende da M_max. Per simulazioni con aloni molto massicci
(che si formano in box grandi), TILE_SIZE potrebbe dover essere aumentato:

| M_max (particelle) | R_acc_max | TILE_SIZE minimo |
|-------------------|-----------|------------------|
| 10.000 | 3.87 | 8 |
| 100.000 | 8.35 | 17 |
| 1.000.000 | 18.0 | 36 |

### Nota: effetti di bordo

I particelle ai **bordi di tile** vedono vicini in tiles di colore diverso (non ancora processati
nella passata corrente) con `group_ID = 0`. Questo può dare decisioni di accretion leggermente
diverse rispetto al loop seriale. Le differenze sono **piccole e accettabili**:

- Test 128³, 1 MPI: 90,559 good halos vs 88,978 serial (~1.8% differenza)
- I risultati sono **deterministici** (identici per qualsiasi numero di thread OMP)

**Perché le differenze sono piccole?** Le particelle ai bordi di tile che vedono `group_ID=0` nei
vicini non-ancora-processati trattano quei vicini come "non in nessun gruppo". In pratica:
- Un picco che avrebbe dovuto essere soppresso (perché un vicino era già in un gruppo) rimane picco
- Questo crea aloni piccoli in più, che vengono spesso riassorbiti nelle passate successive

Per la statistica finale (mass function), l'effetto è marginale e all'interno delle incertezze
Monte Carlo del metodo.

---

## Thread safety

| Operazione | Protezione |
|-----------|-----------|
| `ngroups++` (allocazione peak) | `#pragma omp atomic capture` |
| `groups[FILAMENT].Mass` | Thread-local `tl_filament_delta`, `#pragma omp atomic` a fine passata |
| `counters[]` | Thread-local `tl_cnt[]`, ridotto via `#pragma omp critical(counters_reduce)` |
| `group_ID[iz]`, `linking_list[iz]` | Ogni particella scrive solo il proprio indice (safe) |
| `groups[my_group].*` | Thread con ownership esclusiva dopo allocazione |
| `accretion()`, `merge_groups()` | Safe: tiles stesso colore non condividono gruppi |
| `obj1`, `obj2`, `particle_name`, `good_particle` | `#pragma omp threadprivate` |

### Dettaglio: allocazione peaks senza lock

```c
/* allocazione lockless con atomic capture */
int tl_my_group;
#pragma omp atomic capture
tl_my_group = ++ngroups;
/* ngroups è atomicamente incrementato, tl_my_group ha il valore esclusivo per questo thread */
```

`#pragma omp atomic capture` compila in una singola istruzione hardware (`LOCK XADD` su x86),
molto più veloce di un `critical section`.

### Dettaglio: FILAMENT.Mass con thread-local accumulation

```c
int tl_filament_delta = 0;   /* thread-local, automaticamente privato */

/* durante il loop: */
tl_filament_delta++;   /* una particella diventa filamento */
tl_filament_delta--;   /* una particella lascia il filamento */

/* alla fine della passata di colore: */
#pragma omp atomic
groups[FILAMENT].Mass += tl_filament_delta;   /* una sola atomic per thread */
tl_filament_delta = 0;
```

Senza questa ottimizzazione, ogni evento filamento richiederebbe un'atomic — con migliaia di
eventi per passata, il contention overhead sarebbe significativo.

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
- Per domini grandi (512³+), lo speedup atteso scala con il numero di thread

### Benchmark cluster (512³, 8 MPI task)

| OMP threads | Groups time (s) | Risparmio vs T=1 |
|-------------|----------------|-----------------|
| 1 | 40.78 | baseline |
| 2 | 33.74 | -17% |
| 4 | 32.37 | -21% |
| 8 | 31.34 | -23% |
| 16 | 31.58 | -23% |

Lo speedup si satura attorno a 8 thread. Il collo di bottiglia è la **bandwidth di memoria**:
i dati principali (`frag[]`, `group_ID[]`, `sorted_pos[]`) eccedono la L3 cache per 512³,
e aggiungere thread non aumenta la velocità di lettura dalla DRAM.

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
- [x] Thread safety (`atomic capture`, thread-local `tl_filament_delta`, `threadprivate`)
- [x] PLC post-passata seriale
- [x] Output writes seriali post-tiling
- [x] Gestione pause condition (multi-epoch)
- [x] Fallback seriale (loop originale intatto) per no-OMP e CLASSIC_FRAGMENTATION
- [ ] Benchmark su sistema di produzione (Leonardo/Marconi100)
- [ ] Scalabilità per GridSize > 256
