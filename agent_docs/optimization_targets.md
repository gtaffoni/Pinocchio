# Optimization Targets

> Leggere quando si lavora su performance, profiling, o ottimizzazione.
> Aggiornare con nuovi dati di profiling dopo ogni campagna di misure.

---

## Profiling baseline

Run di riferimento (configurazione e griglia da specificare):

```
Total:                276.131178  (100%)
Initialization:        57.714617  ( 20.90%)
  Density in PS:        7.856252  (  2.85%)
fmax:                 118.474837  ( 42.91%)
  LPT:                 23.623292  (  8.56%)
  Derivatives:         64.058622  ( 23.20%)
    Mem transfer:      10.877061  (  3.94%)
    FFTs:              61.980662  ( 22.45%)
  Collapse times:      30.666932  ( 11.11%)
    inv.collapse:      28.553863  ( 10.34%)
    ellipsoid:          0.000000  (  0.00%)
  Velocities:           0.000000  (  0.00%)
Fragmentation:         99.303349  ( 35.96%)
  Redistribution:      31.655924  ( 11.46%)
  Sorting:              6.661621  (  2.41%)
  Groups:              53.721291  ( 19.45%)
Total I/O:             12.545771  (  4.54%)
```

>ottenuti con questi parametri
BoxSize                500          % physical size of the box in Mpc
GridSize               128          % number of grid points per side
> data del profiling. Questi numeri sono il baseline da battere.

---

## Analisi bottleneck

### Mappa visiva dei tempi

```
FFTs            ████████████████████████  22.5%  → già su GPU (heFFTe) ✅
Groups          ████████████████████      19.5%  → TARGET PRIMARIO 🎯
Initialization  ████████████████████      20.9%  → fuori scope ora
Redistribution  ███████████               11.5%  → TARGET SECONDARIO
Collapse times  ███████████               11.1%  → potenziale futuro
LPT             ████████                   8.6%  → già ottimizzato
I/O             ████                       4.5%  → fuori scope ora
Sorting         ██                         2.4%  → già ottimizzato ✅
Density in PS   ██                         2.9%  → fuori scope ora
```

### Interpretazione

**Groups (53.7s, 19.5%)** — `build_groups.c`
Il loop core di accretion/merging. Seriale per dipendenze causali su `z_c`.
È il bottleneck principale e l'obiettivo del redesign GPU.
Caratteristiche rilevanti per l'ottimizzazione:
- Accessi memoria irregolari (dipende dalla struttura Lagrangiana locale)
- Bassa intensità aritmetica (confronti di distanza, aggiornamento puntatori)
- Dipendenze sequenziali forti tra iterazioni consecutive
- Pattern di accesso: 6 vicini Lagrangiani per particella → potenzialmente
  cache-friendly se le particelle sono ordinate per posizione Lagrangiana

**Redistribution (31.7s, 11.5%)** — `distribute.c`
Comunicazione MPI per redistribuire le particelle tra rank prima del
loop Groups. Costo elevato perché il sort globale per `z_c` mescola
particelle di rank diversi. Un redesign che mantiene la località
Lagrangiana potrebbe eliminare o ridurre drasticamente questa fase.
- Strettamente accoppiato con Groups: un redesign di Groups impatta
  automaticamente Redistribution
- Da analizzare: quanto della Redistribution è inevitabile vs
  artefatto dell'algoritmo attuale

**FFTs (62.0s, 22.5%)** — già su GPU con heFFTe ✅
Non è un target per questo redesign. Già ottimizzato in `refactoring_leonardo`.

**Collapse times (30.7s, 11.1%)** — `collapse_times.c`
Dominato da `inv.collapse` (28.6s). Potenziale target futuro ma
fuori scope del redesign corrente. Da valutare dopo la fragmentazione.

---

## Target di performance

### Target primario — Groups
| Metrica | Baseline | Target | Note |
|---------|----------|--------|------|
| Tempo Groups | 53.7s | < 10s | ~5x speedup su GPU |
| Scaling MPI | scarso | lineare | con domain decomposition |
| GPU utilization | 0% | > 60% | target realistico OpenMP offload |

### Target secondario — Redistribution
| Metrica | Baseline | Target | Note |
|---------|----------|--------|------|
| Tempo Redistribution | 31.7s | < 10s | dipende dal redesign Groups |
| Volume comunicazione MPI | — | ridurre | con località Lagrangiana |

### Target complessivo
| Metrica | Baseline | Target |
|---------|----------|--------|
| Tempo Fragmentazione totale | 99.3s (36%) | < 25s |
| Tempo totale run | 276.1s | < 180s |

---

## Stato ottimizzazioni per modulo

| Modulo | File | Stato | Note |
|--------|------|-------|------|
| FFT | `fmax-heffte.c` | ✅ GPU (heFFTe) | In produzione su Leonardo |
| Sort | `fragment.c` | ✅ Radix sort parallelo | 2.4% — non bottleneck |
| Groups | `build_groups.c` | 🎯 Target | Seriale, 19.5% del totale |
| Redistribution | `distribute.c` | 🎯 Target secondario | MPI overhead, 11.5% |
| Collapse times | `collapse_times.c` | ⏳ Futuro | 11.1%, dopo fragmentazione |
| LPT | `LPT.c` | ✅ Sufficiente | 8.6%, non prioritario |
| I/O | `write_halos.c` | — | 4.5%, fuori scope |

---

## Modello di programmazione GPU

**Scelta: OpenMP offload** (`#pragma omp target`)

Motivazioni:
- Portabilità: stesso codice su NVIDIA (A100/H100), AMD, Intel
- Evita vendor lock-in rispetto a CUDA
- Compatibile con il codebase C esistente
- heFFTe già usa approccio portabile — coerenza architetturale

Pattern di base da usare:
```c
// Esempio struttura OpenMP offload per loop Groups
#pragma omp target teams distribute parallel for \
        map(to: particles[0:N], halo_data[0:N_halos]) \
        map(tofrom: group_assignment[0:N])
for (int i = 0; i < N_batch; i++) {
    // corpo del loop (da definire dopo analisi algoritmica)
}
```

**Prerequisiti per GPU offload della fragmentazione:**
1. Strutture dati flat (no puntatori annidati) — GPU non supporta
   puntatori host in device memory
2. Accessi memoria coalescenti dove possibile
3. Minimizzare trasferimenti host↔device (map solo dati necessari)
4. Algoritmo privo di dipendenze sequenziali (da risolvere nel redesign)

---

## Campagne di profiling da fare

- [ ] Profiling dettagliato di `build_groups.c`: identificare il loop
      critico e misurare intensità aritmetica
- [ ] Profiling `distribute.c`: misurare volume dati comunicati MPI
      e pattern di comunicazione tra rank
- [ ] Scaling test: come variano Groups e Redistribution con
      numero di rank MPI e dimensione griglia?
- [ ] Roofline analysis su GPU target (A100): capire se Groups è
      memory-bound o compute-bound (quasi certamente memory-bound)
- [ ] Aggiornare questo file con i risultati

---

## Note per sessioni Claude Code

Quando si lavora su ottimizzazione, chiedere a Claude Code di:
1. Leggere `build_groups.c` e identificare il loop critico con le
   sue dipendenze dati
2. Analizzare `distribute.c` per capire il pattern di comunicazione MPI
3. Verificare che le strutture dati in `fragment.h` siano compatibili
   con GPU (no puntatori annidati, layout memoria)
4. Proporre refactoring delle strutture dati se necessario prima
   di introdurre OpenMP offload
