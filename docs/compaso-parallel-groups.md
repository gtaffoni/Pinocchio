# CompaSO-Style Parallel Fragmentation e Phase A Pre-Computation per PINOCCHIO

> **Branch:** `feature/compaso-parallel-groups`
> **Base:** `feature/parallel-radix-sort`
> **Autori:** Giuliano Taffoni
> **Data:** Marzo 2026

---

## Indice

- [1. Motivazione](#1-motivazione)
- [2. CompaSO-Style Connected-Component Parallelism](#2-compaso-style-connected-component-parallelism)
  - [2.1 Idea fondamentale](#21-idea-fondamentale)
  - [2.2 Union-Find](#22-union-find)
  - [2.3 Rilevamento delle componenti connesse](#23-rilevamento-delle-componenti-connesse)
  - [2.4 Partizionamento e riordinamento](#24-partizionamento-e-riordinamento)
  - [2.5 Dispatch parallelo con OpenMP](#25-dispatch-parallelo-con-openmp)
  - [2.6 build_groups_component()](#26-build_groups_component)
  - [2.7 Thread safety](#27-thread-safety)
  - [2.8 Fallback sequenziale](#28-fallback-sequenziale)
- [3. Growth Factor Lookup Tables (GFLUT)](#3-growth-factor-lookup-tables-gflut)
  - [3.1 Struttura dati](#31-struttura-dati)
  - [3.2 Interpolazione Catmull-Rom](#32-interpolazione-catmull-rom)
  - [3.3 Interpolazione 2D (SCALE_DEPENDENT)](#33-interpolazione-2d-scale_dependent)
  - [3.4 Funzioni drop-in](#34-funzioni-drop-in)
  - [3.5 Inizializzazione e validazione](#35-inizializzazione-e-validazione)
- [4. Pre-Computation per Particella (Phase A)](#4-pre-computation-per-particella-phase-a)
  - [4.1 Struttura dati](#41-struttura-dati)
  - [4.2 Ciclo di pre-computazione](#42-ciclo-di-pre-computazione)
  - [4.3 virial_fast()](#43-virial_fast)
  - [4.4 Integrazione nel hot path](#44-integrazione-nel-hot-path)
- [5. Riuso degli Array Esistenti](#5-riuso-degli-array-esistenti)
- [6. File Modificati](#6-file-modificati)
- [7. Prestazioni Attese](#7-prestazioni-attese)
  - [7.1 GFLUT](#71-gflut)
  - [7.2 Phase A](#72-phase-a)
  - [7.3 Parallelismo tra componenti](#73-parallelismo-tra-componenti)
  - [7.4 Risultati sperimentali](#74-risultati-sperimentali)
- [8. Limitazioni e Sviluppi Futuri](#8-limitazioni-e-sviluppi-futuri)

---

## 1. Motivazione

La frammentazione di PINOCCHIO costruisce gruppi di materia (aloni, filamenti) processando le particelle in ordine decrescente di $F_{\max}$ (tempo di collasso). Per ogni particella, il codice:

1. Calcola la posizione Euleriana tramite spostamenti LPT (1, 2 e 3 ordine)
2. Valuta fattori di crescita cosmologici $D(z,k)$ tramite interpolazione GSL su spline
3. Calcola il raggio viriale confrontando distanze con i vicini sulla griglia Lagrangiana
4. Decide accrezione su un gruppo esistente o creazione di un nuovo picco

Nel codice originale queste operazioni sono **strettamente sequenziali** e **non thread-safe** (le spline GSL usano stato interno globale). Questo branch introduce tre ottimizzazioni complementari:

- **Parallelismo tra componenti connesse** (ispirato a CompaSO, Hadzhiyska et al. 2022): identifica sottoproblemi indipendenti sulla griglia Lagrangiana e li processa in parallelo con OpenMP
- **Growth Factor Lookup Tables (GFLUT)**: sostituisce le spline GSL con tabelle pre-calcolate su griglia uniforme, eliminando la dipendenza non-thread-safe e riducendo il costo per valutazione
- **Pre-computation per particella (Phase A)**: calcola posizioni Euleriane e $\sigma_D$ in un passo embarrassingly-parallel prima della frammentazione, preparando il codice per future implementazioni GPU

---

## 2. CompaSO-Style Connected-Component Parallelism

### 2.1 Idea fondamentale

Nell'algoritmo di frammentazione, una particella interagisce solo con i suoi **6 vicini face-adjacent** sulla griglia Lagrangiana 3D. Se due gruppi di particelle sono separati da almeno una cella vuota (non collassata o non caricata), non possono mai influenzarsi reciprocamente. Le **componenti connesse** dell'insieme di particelle sulla griglia sono quindi sottoproblemi di frammentazione completamente indipendenti.

```
Griglia Lagrangiana 2D (esempio semplificato):

  . . X X . . X .        Componente A: {X} connesse
  . X X . . X X X        Componente B: {X} connesse
  . . . . . . . .   <--- riga vuota separa A e B
  . . X X X . . .        Componente C: indipendente
  . . . X . . . .
```

### 2.2 Union-Find

L'identificazione delle componenti connesse usa una struttura dati **Disjoint-Set (Union-Find)** con due ottimizzazioni standard:

```c
/* Path-halving find: O(alpha(N)) ammortizzato */
static inline int uf_find(int *parent, int i)
{
  while (parent[i] != i)
    {
      parent[i] = parent[parent[i]];   // path halving
      i = parent[i];
    }
  return i;
}

/* Union-by-rank: l'albero piu' corto punta al piu' alto */
static inline void uf_unite(int *parent, int *rank, int a, int b)
{
  a = uf_find(parent, a);
  b = uf_find(parent, b);
  if (a == b) return;
  if (rank[a] < rank[b]) { int t = a; a = b; b = t; }
  parent[b] = a;
  if (rank[a] == rank[b]) rank[a]++;
}
```

La complessita' ammortizzata per operazione e' $O(\alpha(N))$ dove $\alpha$ e' la funzione inversa di Ackermann — effettivamente $O(1)$ in pratica.

### 2.3 Rilevamento delle componenti connesse

La funzione `find_components()` opera in tre fasi:

1. **Inizializzazione**: ogni particella e' la propria radice (`parent[i] = i`)
2. **Scansione vicini**: per ogni particella, controlla solo i 3 vicini in direzione positiva (+x, +y, +z) per evitare conteggio doppio. Se il vicino esiste nell'insieme di particelle caricate, chiama `uf_unite`
3. **Compressione finale**: `parent[i] = uf_find(parent, i)` per ogni `i`, garantendo che ogni particella punti direttamente alla radice

La ricerca dei vicini usa `find_location()` (lookup diretto $O(1)$ dalla tabella introdotta nel branch `parallel-radix-sort`).

**Riuso memoria**: l'array `indicesY[]` (non piu' necessario dopo l'ordinamento) viene riusato come array `parent[]`. Un array temporaneo `uf_rank[]` viene allocato e liberato al termine.

### 2.4 Partizionamento e riordinamento

La funzione `partition_components()` trasforma l'array compresso `parent[]` in strutture pronte per il dispatch parallelo:

#### Struttura per componente

```c
typedef struct {
  int root;         // radice Union-Find
  int offset;       // offset in particle_order[]
  int count;        // numero di particelle
  int npeaks;       // numero di picchi (per pre-allocazione group ID)
  int group_base;   // primo group ID assegnabile
} component_info;
```

#### Algoritmo

1. **Mapping radice → componente**: usa `group_ID[]` (temporaneamente) come tabella di lookup
2. **Conteggio**: prima passata per contare particelle per componente
3. **Prefix sum**: calcola gli offset nell'array `particle_order[]`
4. **Riempimento**: usa `linking_list[]` (temporaneamente) come cursore; riordina le particelle in modo che ogni componente occupi un segmento contiguo, preservando l'ordine decrescente di $F_{\max}$ all'interno
5. **Conteggio picchi**: per ogni componente, conta quante particelle sono picchi locali (tutti i vicini hanno $F_{\max}$ minore)
6. **group_base**: prefix sum sui conteggi picchi → ogni componente scrive in range non-overlapping di group ID
7. **Ordinamento**: componenti ordinate per dimensione decrescente (insertion sort) per migliorare il bilanciamento del carico

**Riuso memoria**: `group_ID[]` e `linking_list[]` sono usati come buffer temporanei e azzerati prima del ritorno.

### 2.5 Dispatch parallelo con OpenMP

Il ciclo principale di `build_groups()` e' sostituito da:

```c
while (1) {
    /* Determina F_stop: prossimo punto di sincronizzazione
       (redshift di output o pausa segmento) */

    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < par_ncomp; c++) {
        build_groups_component(plist + start, end - start, ...);
    }

    /* Accumula contatori sotto #pragma omp critical */
    /* Aggiorna ngroups globale */
    /* Scrivi output ai punti di sincronizzazione */
    /* Se completato, esci dal loop */
}
```

Scelte di design:

- **`schedule(dynamic, 1)`**: le componenti hanno dimensioni molto variabili (da 1 particella a centinaia di migliaia). Lo scheduling dinamico evita che i thread restino inattivi aspettando la componente piu' grande
- **Punti di sincronizzazione**: il `while` loop si ferma ai redshift di output e ai confini di segmento per scrivere cataloghi. Tra i punti di sync, tutte le componenti avanzano indipendentemente
- **Diagnostica**: viene misurato e stampato il tempo della componente piu' lenta, permettendo di valutare il bilanciamento del carico

### 2.6 build_groups_component()

Funzione autocontenuta che processa la lista di particelle di una singola componente connessa. E' una versione rifattorizzata del loop sequenziale originale con queste differenze:

- **Stato locale**: `my_ngroups`, `my_particle_name`, `my_good_particle`, `local_counters[]`, `local_filament_mass` al posto delle variabili globali
- **Nessun PLC**: la ricostruzione Past Light Cone non e' supportata in modo parallelo
- **Nessuna scrittura output**: l'output e' gestito ai punti di sincronizzazione nel loop esterno
- **Array condivisi**: `group_ID[]`, `linking_list[]`, `groups[]`, `frag[]` sono condivisi ma acceduti in modo thread-safe (vedi sotto)

### 2.7 Thread safety

| Risorsa condivisa | Pattern di accesso | Meccanismo di sicurezza |
|---|---|---|
| `gflut` | Sola lettura dopo init | Inizializzata prima della regione parallela |
| `precomp[]` | Scrittura una volta in init parallelo, poi sola lettura | Scheduling statico in init; sola lettura durante l'uso |
| `group_ID[]` | Scrittura per-particella | Non-overlapping per garanzia di componente connessa |
| `linking_list[]` | Scrittura per-particella | Non-overlapping per garanzia di componente connessa |
| `groups[]` | Scrittura per-gruppo | Range `group_base` non-overlapping per componente |
| `frag[]`, `frag_pos[]` | Sola lettura | Nessuna scrittura durante la frammentazione |
| `counters[]` | Thread-locali, poi accumulati | `#pragma omp critical` per l'accumulo |
| `groups[FILAMENT].Mass` | Accumulato | `#pragma omp critical` per l'accumulo |

L'argomento di correttezza si basa sulla proprieta' fondamentale: **le componenti connesse sulla griglia Lagrangiana non condividono vicini face-adjacent**, quindi nessuna particella nella componente A puo' mai essere vicina di una particella nella componente B.

### 2.8 Fallback sequenziale

Il percorso parallelo e' disabilitato quando:

- E' definito `CLASSIC_FRAGMENTATION` (layout dati diverso)
- Si trova una sola componente connessa (l'intera griglia e' percolante)
- PLC e' attivo (`plc.Fstart > 0`): la ricostruzione richiede sincronizzazione globale ad ogni passo

In tutti questi casi, il loop sequenziale originale viene eseguito senza modifiche.

---

## 3. Growth Factor Lookup Tables (GFLUT)

### 3.1 Struttura dati

```c
#define GFLUT_NPTS 10000

typedef struct {
  int npts;              // = GFLUT_NPTS
  int initialized;       // flag: 0 = non pronto, 1 = pronto
  double z_min, z_max;
  double dz_inv;         // = (npts - 1) / (z_max - z_min)

  double grow1  [NkBINS][GFLUT_NPTS];   // D_1(z,k)    — 1LPT
  double grow2  [NkBINS][GFLUT_NPTS];   // D_2(z,k)    — 2LPT
  double grow31 [NkBINS][GFLUT_NPTS];   // D_31(z,k)   — 3LPT tipo 1
  double grow32 [NkBINS][GFLUT_NPTS];   // D_32(z,k)   — 3LPT tipo 2
  double fomega1[NkBINS][GFLUT_NPTS];   // f*Omega_1   — velocita' 1LPT
  double fomega2[NkBINS][GFLUT_NPTS];   // f*Omega_2   — velocita' 2LPT
  double fomega31[NkBINS][GFLUT_NPTS];  // f*Omega_31  — velocita' 3LPT tipo 1
  double fomega32[NkBINS][GFLUT_NPTS];  // f*Omega_32  — velocita' 3LPT tipo 2
  double hubble [GFLUT_NPTS];           // H(z)

#ifdef SCALE_DEPENDENT
  double kmin, kmax;
#endif
} gflut_data;

static gflut_data gflut = {0};
```

**Footprint di memoria**:
- `SCALE_INDEPENDENT` (`NkBINS=1`): $9 \times 10000 \times 8$ bytes $\approx$ **703 KB**
- `SCALE_DEPENDENT` (`NkBINS=10`): $81 \times 10000 \times 8$ bytes $\approx$ **6.3 MB**

### 3.2 Interpolazione Catmull-Rom

L'interpolazione usa uno schema cubico a 4 punti (Catmull-Rom), che garantisce continuita' $C^1$ senza richiedere la risoluzione di un sistema tridiagonale (come le spline cubiche naturali):

```c
static inline double catmull_rom(const double *y, int npts, int i, double t)
{
  /* Ai bordi: fallback a interpolazione lineare */
  if (i <= 0)
    return y[0] * (1.0 - t) + y[1] * t;
  if (i >= npts - 2)
    return y[npts-2] * (1.0 - t) + y[npts-1] * t;

  double y0 = y[i-1], y1 = y[i], y2 = y[i+1], y3 = y[i+2];

  double a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
  double b =      y0 - 2.5*y1 + 2.0*y2 - 0.5*y3;
  double c = -0.5*y0           + 0.5*y2;
  double d =                y1;

  return ((a*t + b)*t + c)*t + d;   // Horner: 4 multiply + 3 add
}
```

**Proprieta'**:
- Precisione: 4 ordine nell'interno, 2 ordine ai bordi
- Costo: 4 letture + 10 moltiplicazioni + 8 addizioni
- Nessun branching nel percorso interno → ottimale per pipeline CPU e futura portabilita' GPU
- La griglia e' uniforme in $z$, quindi l'indice si calcola con una singola moltiplicazione: `iz = (int)((z - z_min) * dz_inv)`

### 3.3 Interpolazione 2D (SCALE_DEPENDENT)

Per build `SCALE_DEPENDENT` (`NkBINS > 1`), la funzione `gflut_interp()` esegue interpolazione bilineare in $k$ con Catmull-Rom in $z$:

1. Clamping di $z$ a $[z_{\min}, z_{\max}]$
2. Calcolo indice $z$ e parte frazionaria
3. Localizzazione del bin in $k$ tramite $\log_{10}(k)$ sulla griglia uniforme in $k$
4. Due valutazioni Catmull-Rom ai bin $k$ adiacenti
5. Interpolazione lineare in $k$ tra i due valori

Per build `SCALE_INDEPENDENT`, usa direttamente `table[0]` (unico bin).

### 3.4 Funzioni drop-in

Otto funzioni `static inline` sostituiscono le corrispondenti basate su GSL:

| Funzione originale (GSL) | Sostituzione GFLUT | Tabella usata |
|---|---|---|
| `GrowingMode(z, k)` | `fast_GrowingMode(z, k)` | `grow1` |
| `GrowingMode_2LPT(z, k)` | `fast_GrowingMode_2LPT(z, k)` | `grow2` |
| `GrowingMode_3LPT_1(z, k)` | `fast_GrowingMode_3LPT_1(z, k)` | `grow31` |
| `GrowingMode_3LPT_2(z, k)` | `fast_GrowingMode_3LPT_2(z, k)` | `grow32` |
| `fomega(z, k)` | `fast_fomega(z, k)` | `fomega1` |
| `fomega_2LPT(z, k)` | `fast_fomega_2LPT(z, k)` | `fomega2` |
| `fomega_3LPT_1(z, k)` | `fast_fomega_3LPT_1(z, k)` | `fomega31` |
| `fomega_3LPT_2(z, k)` | `fast_fomega_3LPT_2(z, k)` | `fomega32` |
| `Hubble(z)` | `fast_Hubble(z)` | `hubble` |

Le funzioni esistenti (`set_weight`, `virial`, `set_obj_vel`, `condition_for_merging`) controllano `gflut.initialized` e delegano alle versioni fast quando disponibili.

### 3.5 Inizializzazione e validazione

- **`gflut_init(z_min, z_max)`**: chiamata una volta da `fragment()` prima del loop dei segmenti. Campiona tutte le 9 funzioni dalla griglia GSL alla griglia uniforme
- **`gflut_validate()`**: confronta `fast_GrowingMode` vs `GrowingMode` su 1000 valori di $z$ e riporta l'errore relativo massimo. Funzione diagnostica, non interrompe l'esecuzione

---

## 4. Pre-Computation per Particella (Phase A)

### 4.1 Struttura dati

```c
typedef struct {
  PRODFLOAT euler[3];   // posizione Euleriana al proprio Fmax
  double sigmaD;        // sqrt(TrueVariance[S]) * D_1(Fmax-1, k_dens)
} particle_precomp;

static particle_precomp *precomp = NULL;
```

**Footprint**: 20-32 bytes/particella (dipende da `DOUBLE_PRECISION_PRODUCTS`).

### 4.2 Ciclo di pre-computazione

`precompute_particles()` e' un loop **embarrassingly parallel** su tutte le particelle:

```c
#pragma omp parallel for schedule(static)
for (int p = 0; p < subbox.Nstored; p++)
{
  // 1. Decodifica coordinate griglia da frag_pos[p]
  INDEX_TO_COORD(frag_pos[p], ii, jj, kk, subbox.Lgwbl);

  // 2. Converti Fmax in redshift
  double z = (double)frag[p].Fmax - 1.0;

  // 3. Calcola pesi LPT tramite fast_set_weight
  fast_set_weight(&data, z, k_dens);

  // 4. Posizione Euleriana: q + D1*psi1 + D2*psi2 + D31*psi31 + D32*psi32
  for (d = 0; d < 3; d++)
    precomp[p].euler[d] = q2x(p, d, &data);

  // 5. sigmaD per il raggio viriale
  precomp[p].sigmaD = sqrt(TrueVariance[S]) * fast_GrowingMode(z, k_dens);
}
```

**Ciclo di vita**:
- `precomp_allocate(Nstored)` → prima di ogni `build_groups()`
- `precompute_particles()` → riempie l'array in parallelo
- `precomp_free()` → dopo `build_groups()`

### 4.3 virial_fast()

Funzione puramente algebrica che prende $\sigma_D$ pre-calcolato:

```c
static inline double virial_fast(int mass, double sigmaD, int flag)
{
  double rv = sigmaD * ScaleFactor(mass);    // raggio viriale
  double rr = rv * rv;
  if (flag)
    rr *= RVIR_FACTOR;
  return rr;
}
```

Nessuna chiamata a funzioni cosmologiche → eliminazione completa delle spline GSL dal loop interno.

### 4.4 Integrazione nel hot path

In `condition_for_accretion()`, per le chiamate 1, 2 e 3 (la stragrande maggioranza):

**Prima** (codice originale):
```c
set_point(&obj, ind);          // 4-12 valutazioni growth factor
for (d = 0; d < 3; d++)
  x[d] = q2x(ind, d, &obj);   // 3 valutazioni displacement
rr = virial(mass, Fmax, 1);   // 1+ valutazione growth factor
d2 = dist2(x, center);        // 3 sottrazioni + 3 moltiplicazioni
```

**Dopo** (con pre-computation):
```c
rr = virial_fast(mass, precomp[ind].sigmaD, 1);   // pura aritmetica
for (d = 0; d < 3; d++) {
  dd = precomp[ind].euler[d] - center[d];          // lettura array
  d2 += dd * dd;
  if (d2 >= rr) break;                             // early exit
}
```

L'early exit dimensione-per-dimensione evita calcoli inutili per particelle chiaramente fuori dal raggio viriale.

La **chiamata 4** (ri-accrezione filamento) usa il percorso originale perche' valuta la particella a un $F_{\max}$ diverso dal proprio.

---

## 5. Riuso degli Array Esistenti

L'implementazione minimizza le allocazioni aggiuntive riusando array gia' presenti:

| Array | Uso originale | Uso temporaneo | Ripristino |
|---|---|---|---|
| `indicesY[Nalloc]` | Non usato dopo il sort | Array `parent[]` per Union-Find | Sovrascritto (non piu' necessario) |
| `group_ID[Nalloc]` | Assegnamento gruppi | Mapping `root_to_comp[]` in `partition_components` | Azzerato con `memset` |
| `linking_list[Nalloc]` | Lista di linking | Array cursore in `partition_components` | Azzerato dopo l'uso |

**Nuove allocazioni**:

| Struttura | Dimensione | Durata |
|---|---|---|
| `gflut_data gflut` | ~703 KB - 6.3 MB (statica) | Intera fase di frammentazione |
| `particle_precomp *precomp` | 20-32 B/particella | Un segmento di frammentazione |
| `component_info *par_comps` | ~20 B/componente | Prima chiamata di `build_groups` |
| `int *par_particle_order` | 4 B/particella | Prima chiamata di `build_groups` |
| `int *par_cursor` | 4 B/componente | Prima chiamata di `build_groups` |
| `int *par_ngroups_created` | 4 B/componente | Prima chiamata di `build_groups` |
| `int *uf_rank` (temporaneo) | 4 B/particella | Solo durante `find_components` |

---

## 6. File Modificati

### Rispetto al branch `feature/parallel-radix-sort`

| File | Tipo | Modifiche |
|---|---|---|
| `src/build_groups.c` | Modificato (~1200 righe nuove) | Union-Find, `find_components`, `partition_components`, dispatch parallelo, `build_groups_component`, GFLUT completo, Phase A pre-computation, `virial_fast`, `fast_set_weight`, modifiche a `condition_for_accretion`, `condition_for_merging`, `set_obj_vel`, `virial` |
| `src/fragment.c` | Modificato (+11 righe) | Chiamate a `gflut_init`/`gflut_validate` prima del loop, `precomp_allocate`/`precompute_particles` prima di `build_groups`, `precomp_free` dopo |
| `src/fragment.h` | Modificato (+7 righe) | Prototipi: `gflut_init`, `gflut_validate`, `precomp_allocate`, `precomp_free`, `precompute_particles` |
| `src/pinocchio.h` | Modificato (+6 righe) | Stessi prototipi + aggiornamento dichiarazioni `condition_for_accretion`, `condition_for_merging`, `merge_groups`, `accretion` con parametro `pos_data*` |

### Nessun file nuovo

Tutto il codice e' contenuto nei file esistenti. Non sono stati aggiunti nuovi file sorgente.

---

## 7. Prestazioni Attese

### 7.1 GFLUT

Ogni chiamata a `GrowingMode(z,k)` nel codice originale invoca la valutazione di spline GSL: binary search $O(\log N_{\text{knots}})$ + polinomio cubico + potenziali cache miss sui dati interni GSL. La sostituzione GFLUT:

- Calcolo indice diretto: `iz = (int)((z - z_min) * dz_inv)` → $O(1)$
- Interpolazione Catmull-Rom: 4 letture contigue + 10 MUL + 8 ADD
- Accesso memoria prevedibile e contiguo

**Speedup stimato per singola chiamata**: 3-10x a seconda dello stato della cache.

### 7.2 Phase A

Per le chiamate 1, 2 e 3 di `condition_for_accretion` (la maggioranza):

| Operazione | Prima | Dopo |
|---|---|---|
| Calcolo posizione | `set_weight` (4-12 growth eval) + `q2x` (3 displacement) | 3 letture da `precomp[].euler[]` |
| Raggio viriale | `virial()` (1+ growth eval) | `virial_fast()` (pura aritmetica) |
| Test distanza | `dist2()` completo | Early exit per dimensione |

**Impatto**: le funzioni di crescita vengono chiamate milioni-miliardi di volte durante la frammentazione. La pre-computation le elimina completamente dal loop interno per ~75% delle chiamate.

### 7.3 Parallelismo tra componenti

| Scenario | Speedup OpenMP | Note |
|---|---|---|
| Molte componenti piccole (universo giovane, collasso sparso) | Quasi-lineare con $P$ thread | Caso ottimale |
| Una componente dominante + molte piccole | Limitato dalla componente piu' grande | Legge di Amdahl |
| Singola componente percolante | Nessuno (fallback sequenziale) | Overhead trascurabile (solo Union-Find) |

L'overhead della fase Union-Find + partizionamento e' $O(N_{\text{stored}} \cdot \alpha(N_{\text{stored}}))$ — essenzialmente $O(N)$ in pratica — trascurabile rispetto al loop di frammentazione.

### 7.4 Risultati sperimentali

Test su run `example` (~400K particelle totali):

**Diagnostica componente dominante (np=8, OMP_NUM_THREADS=1)**:
```
Connected components: 2535 (largest: 107707, smallest: 1, total peaks: 18178)
Parallel step: wall=0.0101s, slowest component #0 (9851 particles, 0.0100s = 99.1% of wall)
```

**Scaling OpenMP (np=1)**:

| OMP_NUM_THREADS | Groups (s) | Sorting (s) |
|---|---|---|
| 1 | 0.699 | 0.077 |
| 2 | 0.702 | 0.072 |
| 4 | 0.700 | 0.073 |
| 8 | 0.699 | 0.073 |

**Scaling MPI**:

| np | Groups (s) | Componente max | Note |
|---|---|---|---|
| 1 | 0.699 | 361256 particelle | Intero dominio |
| 2 | 1.223 | (piu' grande) | Decomposizione sfavorevole |
| 4 | 0.660 | (piu' piccola di np=1) | Dominio locale ridotto |
| 8 | 0.375 | 107707 particelle | 1.9x vs np=1 |

La componente piu' grande occupa il 99%+ del tempo wall-clock in tutti i casi, rendendo il parallelismo OpenMP tra componenti inefficace per questo run di test. Il beneficio principale viene dalla **decomposizione MPI** che riduce la dimensione della componente dominante.

---

## 8. Limitazioni e Sviluppi Futuri

### Limitazioni correnti

1. **PLC non supportato in parallelo**: la ricostruzione Past Light Cone richiede sincronizzazione globale incompatibile con il dispatch per-componente
2. **Componente dominante**: in simulazioni a basso redshift con alta densita', una singola componente percolante puo' contenere la maggior parte delle particelle, annullando il beneficio del parallelismo OpenMP
3. **`CLASSIC_FRAGMENTATION`**: il percorso parallelo non e' attivo per questa modalita'

### Sviluppi futuri

1. **GPU offload (Phase B)**: le tabelle GFLUT e l'array `precomp[]` sono progettati per essere copiati su GPU. Il loop di `precompute_particles()` e' un kernel CUDA/OpenMP-target naturale
2. **Parallelismo intra-componente**: per la componente dominante, possibili approcci includono:
   - Decomposizione spaziale della componente in sotto-blocchi
   - Pipeline: processare batch di particelle con pre-fetch parallelo
3. **heFFTe per collapse times**: integrazione con la libreria heFFTe per il calcolo parallelo GPU dei tempi di collasso (gia' prototipato nel branch `feature-parallel-GPU-mem-copy`)
4. **Scaling MPI**: con piu' task MPI, il dominio locale si riduce e la componente massima diventa piu' piccola, migliorando naturalmente il bilanciamento del carico tra componenti
