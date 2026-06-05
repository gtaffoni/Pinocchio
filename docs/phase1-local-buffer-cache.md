# Phase 1: Local Buffer Cache per Tile

**Branch:** `feature/omp_parallel_groups`
**File modificato:** `src/build_groups.c`
**Data:** 2026-06-05

---

## Motivazione

Il tiling 8-colori è implementato e funzionante, ma non scala linearmente con il numero di thread OpenMP per la fase Groups. La causa è il **cache thrashing** indotto dagli accessi random alle strutture dati globali.

### Accessi per ogni vicino lagrangiano (6 per particella)

Per ogni particella nel tile, il loop sui 6 vicini esegue:

```c
tl_pos = find_location(tl_i1, tl_j1, tl_k1);
// find_location fa:
//   gpos = COORD_TO_INDEX(i,j,k,Lgwbl)   → accesso a Lgwbl (L1)
//   return sorted_pos[gpos]               → accesso RANDOM a sorted_pos (1 MB)

tl_neigh[nn] = group_ID[tl_pos];          // accesso RANDOM a group_ID (1 MB)
frag[tl_iz].Fmax > frag[tl_pos].Fmax      // accesso RANDOM a frag (14 MB)
```

Con `subbox.Lgwbl ≈ 128³`:
- `sorted_pos[]`: ~2 MB → non sta in L1/L2, miss rate alto
- `group_ID[]`: ~2 MB → stesso problema
- `frag[]`: ~14 MB → sicuramente DRAM per ogni accesso

Con più thread in parallelo sui propri tile, i thread si contendono le stesse linee di cache (false sharing sulle strutture random-access) → **scaling piatto**.

### Misurazione baseline (128³, 1 MPI task)

| Threads | Groups time |
|---------|------------|
| 1       | 0.689s     |
| 4       | 0.677s     |
| 8       | ~0.680s    |

Speedup misurato: ~1.02x con 4 thread. Quasi nullo.

---

## Soluzione: buffer locale per tile

Prima di processare ogni tile, si copia il tile + la shell di 1 cella adiacente in due buffer compatti allocati sullo stack del thread OpenMP:

- `tl_local_gid[LS3]`  — `group_ID` per ogni posizione del buffer
- `tl_local_fmax[LS3]` — `frag[].Fmax` per ogni posizione del buffer

Per `TILE_SIZE=8`: `LS = 10`, `LS3 = 1000` → buffer = 1000 × (4+4) byte = **8 KB → entra in L1 cache**.

Le 6 letture per vicino diventano accessi al buffer locale (L1 hit) invece di random access alla DRAM.

---

## Implementazione

### Costanti

```c
#define TILE_SIZE 8
#define LS   (TILE_SIZE + 2)    // 10: lato del buffer locale
#define LS2  (LS * LS)          // 100
#define LS3  (LS * LS * LS)     // 1000
#define LIDX(li,lj,lk) ((li)*LS2 + (lj)*LS + (lk))
```

La dimensione `LS = TILE_SIZE + 2` copre il tile stesso (indici locali `[1, TILE_SIZE]`) più la shell di 1 cella su ogni lato (indice `0` e `TILE_SIZE+1`).

### Mappatura coordinate

Per una particella al tile che inizia in `(tile_sx, tile_sy, tile_sz)`:

| Posizione griglia globale | Indice locale `li` |
|---------------------------|--------------------|
| `tile_sx - 1`             | 0                  |
| `tile_sx`                 | 1                  |
| `tile_sx + k`             | `k + 1`            |
| `tile_sx + TILE_SIZE - 1` | `TILE_SIZE`        |
| `tile_sx + TILE_SIZE`     | `TILE_SIZE + 1`    |

Per la particella corrente a `(tl_ibox, tl_jbox, tl_kbox)`:
```c
int tl_li_own = tl_ibox - tile_sx + 1;   // ∈ [1, TILE_SIZE]
int tl_lj_own = tl_jbox - tile_sy + 1;
int tl_lk_own = tl_kbox - tile_sz + 1;
```

Per il vicino in direzione `±x`:
```c
// nn=0 (−x): li = tl_li_own - 1
// nn=1 (+x): li = tl_li_own + 1
```

Tutti i vicini hanno indici locali in `[0, LS-1]` (mai fuori range).

### `fill_local_buffer()`

```c
static inline void fill_local_buffer(
    int sx, int sy, int sz,
    int *tl_local_gid, PRODFLOAT *tl_local_fmax)
{
    for (li = 0; li < LS; li++) {
        gi = sx - 1 + li;
        // gestione PBC / out-of-domain per dimensione x
        for (lj = 0; lj < LS; lj++) {
            gj = sy - 1 + lj;
            // gestione PBC / out-of-domain per dimensione y
            for (lk = 0; lk < LS; lk++) {
                gk = sz - 1 + lk;
                // gestione PBC / out-of-domain per dimensione z
                int lidx = LIDX(li, lj, lk);
                int fp = sorted_pos[COORD_TO_INDEX(gii, gjj, gkk, Lgwbl)];
                if (fp >= 0) {
                    tl_local_gid [lidx] = group_ID[fp];
                    tl_local_fmax[lidx] = frag[fp].Fmax;
                } else {
                    tl_local_gid [lidx] = 0;
                    tl_local_fmax[lidx] = 0.0f;
                }
            }
        }
    }
}
```

**Invariante:** `tl_local_fmax[lidx] > 0` se e solo se esiste una particella collassata a quella posizione griglia (`sorted_pos[gpos] >= 0`). Usato per distinguere "nessuna particella" da "particella non ancora processata (group_ID=0)".

### Loop 6-vicini modificato

```c
// Before switch: compute own buffer coords once
int tl_li_own = tl_ibox - tile_sx + 1;
int tl_lj_own = tl_jbox - tile_sy + 1;
int tl_lk_own = tl_kbox - tile_sz + 1;

for (tl_nn = 0; tl_nn < NV; tl_nn++) {
    int tl_lidx;
    switch (tl_nn) {
    case 0: tl_lidx = LIDX(tl_li_own-1, tl_lj_own, tl_lk_own); /* −x */
            tl_i1 = (pbc_x && tl_ibox==0 ? Lgwbl[x]-1 : tl_ibox-1);
            tl_j1 = tl_jbox; tl_k1 = tl_kbox; break;
    // ... altri 5 casi
    }

    int tl_ngid        = tl_local_gid [tl_lidx];  // L1 hit
    PRODFLOAT tl_nfmax = tl_local_fmax[tl_lidx];  // L1 hit

    tl_neigh[tl_nn] = tl_ngid;
    if (tl_nfmax > 0)                              // particella esiste?
        tl_peak_cond &= (frag[tl_iz].Fmax > tl_nfmax);

    if (tl_ngid == FILAMENT) {
        tl_neigh[tl_nn] = 0;
        tl_pos = find_location(tl_i1, tl_j1, tl_k1);  // solo per FILAMENT
        tl_fil_list[tl_nf][...] = ...;
        tl_nf++;
    }
}
```

**Nota:** `find_location()` viene chiamato solo nel path FILAMENT (raro), non per ogni vicino.

**Nota sulla condizione peak:** la condizione `tl_nfmax > 0` replica esattamente il comportamento originale `tl_pos >= 0`:
- Nessuna particella collassata (`sorted_pos < 0`): `fmax=0` → skip, sia prima che dopo
- Particella esistente ma non ancora processata (`group_ID=0`, `fmax>0`): check Fmax eseguito, sia prima che dopo
- Particella processata (`group_ID>0`, `fmax>0`): check Fmax eseguito, sia prima che dopo

---

## Il bug merge_groups e la fix

### Cos'è `merge_groups`

Quando una particella ha due gruppi vicini abbastanza vicini tra loro, li fonde. `merge_groups(grp_large, grp_small)` esegue:

```c
// Aggiorna group_ID di tutti i particle in grp_small → grp_large
i1 = groups[grp_small].point;
while (linking_list[i1] != groups[grp_small].point) {
    group_ID[i1] = grp_large;
    i1 = linking_list[i1];
}
group_ID[i1] = grp_large;

// Concatena le linked list circolari
linking_list[groups[grp_large].bottom] = groups[grp_small].point;
linking_list[groups[grp_small].bottom] = groups[grp_large].point;
groups[grp_large].bottom = groups[grp_small].bottom;

// Invalida grp_small
groups[grp_small].point  = -1;
groups[grp_small].bottom = -1;
```

### Il bug: buffer stale dopo merge

Lo scenario problematico, all'interno di un singolo tile:

```
Particella P1 (Fmax=10) → PEAK → group_ID[P1] = gA
  buffer update: tl_local_gid[P1_slot] = gA

Particella P2 (Fmax=9)  → PEAK → group_ID[P2] = gB
  buffer update: tl_local_gid[P2_slot] = gB

Particella P3 (Fmax=8)  → vicini {gA, gB} → MERGE
  merge_groups(gA, gB):
    → group_ID[P2] = gA  (P2 era in grp_B)      ← aggiornato nel global array
    → gB.point = -1  (invalidato)
  tl_neigh[...] aggiornato: gB → gA
  tl_local_gid[P2_slot] = gB                    ← STALE! non aggiornato

Particella P4 (Fmax=7)  → vicino è P2
  legge tl_local_gid[P2_slot] = gB  ← gruppo INVALIDO
  tl_neigh = gB
  condition_for_accretion(gB) → accretion(gB, ...)
  accretion() accede groups[gB].point = -1
  merge_groups() cerca linking_list[i] == groups[gB].point = -1
  → il loop non termina mai → "ERROR: infinite loop in merge_groups"
```

### La fix: sync del buffer dopo ogni merge

```c
// Dopo merge_groups(large, small) + aggiornamento tl_neigh:
int tl_bk;
for (tl_bk = 0; tl_bk < LS3; tl_bk++)
    if (tl_local_gid[tl_bk] == tl_small)
        tl_local_gid[tl_bk] = tl_large;
```

Costo: 1000 confronti interi → ~1 ns su hardware moderno (tutto in L1). Il buffer non cresce mai oltre LS3, quindi il costo è **O(1) costante** indipendentemente dalla dimensione del problema.

### Sync aggiuntivo: filament accretion

Analogamente, quando una particella FILAMENT viene accretata in `accretion()`, il suo `group_ID` globale cambia da `FILAMENT` a `tl_to_group`. Aggiornamento del buffer:

```c
int tl_fli = tl_fil_list[tl_ifil][0] - (tile_sx - 1);
int tl_flj = tl_fil_list[tl_ifil][1] - (tile_sy - 1);
int tl_flk = tl_fil_list[tl_ifil][2] - (tile_sz - 1);
if (tl_fli >= 0 && tl_fli < LS && ...)
    tl_local_gid[LIDX(tl_fli, tl_flj, tl_flk)] = tl_to_group;
```

Le particelle filament che sono vicine della particella corrente sono per costruzione dentro il buffer (sono a distanza ≤1 dalla particella, che è nel tile → sono nella shell del buffer).

---

## Punti di sincronizzazione del buffer

Il buffer `tl_local_gid` viene modificato in 3 punti:

| Evento | Posizione nel codice | Descrizione |
|--------|---------------------|-------------|
| `fill_local_buffer()` | Inizio di ogni tile | Snapshot iniziale da global array |
| Per-particle update | Fine di ogni particella | `tl_local_gid[own_pos] = group_ID[tl_iz]` |
| Post-merge sync | Dopo ogni `merge_groups` | Scan LS3, replace `grp_small → grp_large` |
| Post-filament-accretion | Dopo ogni filament accretato | Update singolo slot |

`tl_local_fmax` **non viene mai modificato** dopo il fill: i valori Fmax sono read-only durante la frammentazione.

---

## Cosa non cambia

- Schema 8 colori: invariato
- `accretion()`, `merge_groups()`: invariate (scrivono sul global array)
- `linking_list[]`, `groups[]`: invariati (non localizzati)
- Path `CLASSIC_FRAGMENTATION`: non usa il tiling, non toccato
- Build senza OMP (`make`): tutto il codice del buffer è dentro `#ifdef _OPENMP`, nessun impatto

---

## Correttezza verificata

| Configurazione | Good halos | Errori |
|---------------|-----------|--------|
| Baseline tiling (1 MPI, 1 OMP) | 90559 | nessuno |
| Baseline tiling (1 MPI, 4 OMP) | 90559 | nessuno |
| Phase 1 buffer (1 MPI, 1 OMP) | 90559 | nessuno |
| Phase 1 buffer (1 MPI, 4 OMP) | 90559 | nessuno |
| Phase 1 buffer (4 MPI, 4 OMP) | 90580 | nessuno |

Tutti identici al baseline. Nessun `merge_groups infinite loop`.

---

## Performance: 128³ (risultato atteso)

Su 128³, il buffer non porta benefici misurabili: il working set (sorted_pos + group_ID + frag per ~128³ particelle) rientra nelle cache L2/L3 del host macOS. Il beneficio reale è atteso su **512³+** dove queste strutture spill nella DRAM (sorted_pos: ~8 MB, group_ID: ~8 MB, frag: ~112 MB per 512³).

Il beneficio fondamentale rimane strutturale: ogni tile processa i propri vicini da 8 KB di buffer L1 invece che da ~128 MB di DRAM. Con più thread che processano tile diversi in parallelo, il working set per thread rimane sempre in L1 indipendentemente dalla dimensione del problema.

---

## Prossimi passi

- **Phase 2**: Localizzare `linking_list[]` nel buffer (per eliminare gli accessi random durante `accretion()`)
- **Phase 3**: Port GPU — il buffer locale è la forma embrionale della `volume_data` GPU-friendly (shared memory per warp)
