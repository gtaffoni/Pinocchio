# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PINOCCHIO Redesign — Multi-Agent Workspace

PINOCCHIO è un codice cosmologico che genera cataloghi di aloni tramite
Lagrangian Perturbation Theory (LPT) + ellipsoidal collapse + frammentazione.
Opera in tre fasi pipeline:

1. **IC Generation** — versione modificata di GenIC integrata
2. **Collapse time computation** — già ottimizzata con heFFTe (multi-backend GPU)
3. **Fragmentation** — loop seriale per task MPI, **bottleneck principale**

### Modifiche già integrate
- heFFTe per FFT nella fase 2 (CUDA/ROCm/oneMKL backend)
- Radix sort al posto di qsort per sorting per z_c decrescente

### Target del redesign
- Merge di `src/` e `src_leonardo/` in codebase unificata
- Frammentazione accelerata: MPI + OpenMP (threading CPU)
- Frammentazione su GPU: OpenMP 5.x target offload
- Portabilità multi-piattaforma (no vendor lock-in, come heFFTe)
- Scale target: >4096³ particelle (regime exascale)
- Compilatori: clang (primario) + nvcc dove necessario

### Build system
Makefile esistente. Non migrare a CMake senza istruzione esplicita.
Il Makefile deve continuare a funzionare dopo ogni modifica.

---

## Build e compilazione

### Compilazione standard (docker/linux base)
Sei in MacOS per cui per compilare devi eseguire uno container docker da uno script 
che trovi nella directory Docker

Docker/docker-run.sh 

```bash
cd src/
# Default SYSTYPE=docker; per altri sistemi:
../Docker/docker-run.sh make SYSTYPE=base          # generico linux con env vars FFTW_LIB etc.
../Docker/docker-run.sh make SYSTYPE=Marconi100    # Marconi100 (CINECA)
../Docker/docker-run.sh make SYSTYPE=pleiadi       # cluster Pleiadi
../Docker/docker-run.sh make SYSTYPE=LeonardoBoost # Leonardo (CINECA), usa nvc

Dato che sei in docker devi usare il default

../Docker/docker-run.sh make

# Debug build:
../Docker/docker-run.sh make DEBUG=YES

# Con OpenMP threading per FFT:
../Docker/docker-run.sh make OMP=YES

# Clean:
../Docker/docker-run.sh make clean
```

### Dipendenze richieste
- MPI (OpenMPI o simile)
- FFTW3 con supporto MPI (e opzionalmente OpenMP)
- PFFT (Parallel FFT)
- GSL (GNU Scientific Library)


### Eseguire il codice

```bash
# Esempio con 5 task MPI nella directory example/
cd example/
../Docker/docker-run.sh mpirun --allow-run-as-root -np 4 ../src/pinocchio.x parameter_file

# Per generare solo le ICs (modo 3):
../Docker/docker-run.sh mpirun --allow-run-as-root -np N ../src/pinocchio.x parameter_file 3
```

### Verifica output di riferimento

```bash
# Confronto statistico con i cataloghi di riferimento src_Leonardo (RACCOMANDATO)
# I cataloghi di riferimento sono in state/reference_catalogs/ (generati da src_Leonardo, 4 MPI)
python scripts/compare_catalogs.py example/ [--redshift 0.0] [--mf] [--verbose]

# Confronto visivo con output atteso (PlotExample.py nella directory example/)
cd example/
python PlotExample.py   # genera mf.png, lss.png, plc.png

# Per rigenerare i cataloghi di riferimento da src_Leonardo:
./scripts/generate_reference_catalogs.sh [NP]
```

---

## Validazione correttezza e benchmark prestazioni

### Regola fondamentale

Ogni volta che si modifica il codice (ottimizzazioni, refactoring, nuove feature), occorre:
1. Eseguire il run di riferimento (`example/`, 4 task MPI, parametri standard)
2. Verificare che i **valori statistici** coincidano con i riferimenti sotto
3. Misurare il **timing** e confrontarlo con il baseline

### Valori statistici di riferimento (invarianti fisici)

Questi numeri **non devono cambiare** tra il codice originale e quello ridisegnato,
a parità di `parameter_file` e `RandomSeed`.  Piccole differenze sono tollerate ma devono essere 
indicate in modo chiaro.

```
# Con 1 MPI task (seriale, valore canonico):
Total number of peaks:                 107684
Total number of good halos:            88981
Particles with N neighbouring groups:  331378 117995 13584 430 4 0
Total number of accretion events:      268658
Accretion before evaluating merger:    79071
Accretion after evaluating merger:     364
Accretion of filament particles:       48327

# Con 4 MPI task (riferimento src_Leonardo, cataloghi in state/reference_catalogs/):
Total number of good halos:            88982  # +1 per boundary MPI
```

> **Nota:** questi valori sono per `example/parameter_file` con `GridSize=128`, `RandomSeed=486604`.
> La differenza 88981 vs 88982 è dovuta alla decomposizione MPI del dominio (1 particella boundary
> classificata diversamente). Per confronto preciso dei cataloghi usare `compare_catalogs.py`.

### Timing di riferimento (baseline prestazioni)

Run su sistema di riferimento (da aggiornare con sistema/configurazione usata):

```
Total               :       9.060099
Initialization      :       0.441936 ( 4.88%)
  Density in PS     :       0.340667 ( 3.76%)
fmax                :       4.802927 (53.01%)
  LPT               :       0.814490 ( 8.99%)
  Derivatives       :       1.541697 (17.02%)
    Mem transfer    :       0.144739 ( 1.60%)
    FFTs            :       1.659320 (18.31%)
  Collapse times    :       2.390869 (26.39%)
  Velocities        :       0.000000 ( 0.00%)
Fragmentation       :       3.768899 (41.60%)
  Redistribution    :       0.795299 ( 8.78%)
  Sorting           :       0.863042 ( 9.53%)
  Groups total      :       2.447627 (27.02%)
  Groups PLC        :       1.073399 (11.85%)
Total I/O           :       0.078045 ( 0.86%)
```

I risultati di benchmark comparativi vanno registrati in `state/benchmark_log.md`.
> **Nota:** questi valori sono per `example/parameter_file` con `GridSize=128`, `RandomSeed=486604` e
> con 1 task mpi.  Se si cambiano i parametri del parameter file i valori cambieranno — ma confrontati 
> tra vecchio e nuovo codice mostrano eventuali miglioramenti.

### Docker

```bash
# Build e run container Docker
cd Docker/
bash docker-run.sh
```

---

## Struttura sorgenti

```
src/            # PINOCCHIO corrente (heFFTe + radix sort integrati)
src_Leonardo/   # versione Leonardo da mergere (ha fmax-heffte.c, GPU collapse)
agents/         # system prompt degli agenti specializzati
state/          # stato condiviso tra agenti
docs/           # documentazione generata (Doxygen)
example/        # parametri e output di riferimento per 128^3 box
tests/          # test di validazione HMF e P(k)
scripts/        # ReadPinocchio5.py per leggere output binario
```

### File sorgente chiave in `src/`

| File | Ruolo |
|------|-------|
| `pinocchio.c` | Entry point, orchestrazione pipeline |
| `pinocchio.h` | Header globale (include MPI, FFTW, PFFT, GSL, OpenMP) |
| `fmax.c` | Calcolo F=1+z_c per ogni particella e smoothing radius |
| `fmax-pfft.c` | Backend FFT via PFFT per fmax |
| `fragment.c` | **Fragmentation** — bottleneck, loop seriale |
| `build_groups.c` | Costruzione degli aloni e merger history |
| `parallel_sort.c/h` | Radix sort parallelo (sostituisce qsort) |
| `gpu_offload.c/h` | Infrastruttura per GPU offload (in sviluppo) |
| `collapse_times.c` | Calcolo tempi di collasso ellissoidale |
| `LPT.c` | Displacement fields 2LPT e 3LPT |
| `distribute.c` | Distribuzione dati tra task MPI |
| `allocations.c` | Gestione memoria |
| `variables.c` | Variabili globali |
| `GenIC.c` | Generazione initial conditions |
| `initialization.c` | Setup iniziale della simulazione |
| `cosmo.c` | Calcoli cosmologici (distanze, growth factor, etc.) |
| `ReadParamfile.c` | Parsing del parameter file |

### Differenze `src/` vs `src_Leonardo/`
`src_Leonardo/` contiene `fmax-heffte.c` (backend heFFTe per GPU), varianti GPU del calcolo collapse times (`collapse_times_GPU.c`), e la versione `fragment.new.c`. Il merge di questi è il task prioritario.

---

## Agenti disponibili e modelli

| Agente | File system prompt | Modello | Ruolo |
|--------|-------------------|---------|-------|
| cosmologist | agents/cosmologist.md | claude-sonnet-4-20250514 | Algoritmi, fisica, literature search |
| hpc_expert | agents/hpc_expert.md | claude-sonnet-4-20250514 | Implementazione C, MPI, OpenMP |
| gpu_expert | agents/gpu_expert.md | claude-sonnet-4-20250514 | OpenMP offload, CUDA, ottimizzazione device |
| reviewer | agents/reviewer.md | claude-opus-4-20250514 | Code review critico, correttezza fisica |
| documenter | agents/documenter.md | claude-haiku-4-5-20251001 | Doxygen, README, commenti |

---

## Workflow obbligatorio per ogni modifica al codice

### Regola fondamentale: Git branch per ogni approccio algoritmico

```
main
├── merge/unify-src              # merge src/ + src_leonardo/
├── feat/fragmentation-openmp    # parallelizzazione OpenMP interna al rank
├── feat/fragmentation-unionfind # metodo union-find parallelo (se proposto)
├── feat/fragmentation-approx-X  # qualsiasi metodo alternativo dalla letteratura
└── feat/gpu-offload-frag        # OpenMP target offload della frammentazione
```

**MAI committare direttamente su main.**
**Ogni proposta algoritmica alternativa = nuovo branch da main.**

### Sequenza per una nuova feature

1. `git checkout main && git checkout -b feat/nome-feature`
2. **Cosmologist agent** — proposta algoritmica con motivazione fisica
3. **HPC Expert agent** — implementazione in C
4. **GPU Expert agent** — porting OpenMP offload (se applicabile)
5. **Reviewer agent** (Opus) — review critica, max 3 iterazioni di correzione
6. **Documenter agent** (Haiku) — documentazione Doxygen
7. Test di compilazione con Makefile esistente
8. Benchmark vs branch main → risultati in `state/benchmark_log.md`
9. PR verso main solo dopo review APPROVED

### Sequenza per il merge iniziale (prima cosa da fare)

1. HPC Expert analizza differenze tra `src/` e `src_leonardo/`
2. Produce report delle differenze in `state/merge_analysis.md`
3. Implementa merge preservando Makefile
4. Reviewer valida che la pipeline funzioni end-to-end
5. Documenter aggiorna header e commenti

---

## Stato condiviso

Prima di iniziare qualsiasi task, leggi:
- `state/current_task.md` — task corrente assegnato dall'orchestratore
- `state/benchmark_log.md` — risultati di performance comparativi
- `state/merge_analysis.md` — analisi differenze src/ vs src_leonardo/
- `state/review_log.md` — storico delle review con esito

Dopo ogni task, aggiorna lo stato rilevante.

---

## Fisica di riferimento (contesto per tutti gli agenti)

### Algoritmo PINOCCHIO — sintesi

**Orbit Crossing (OC):**
- Campo di densità ρ(q) su griglia cubica, q = coordinate lagrangiane
- Potenziale φ(q,R) calcolato con FFT per ~20 smoothing radii R logaritmicamente spaziati
- Collapse quando det|∂x/∂q| = 0 (Jacobiano nullo)
- Per ogni particella: si registra z_c massimo, R_c corrispondente, velocità Zel'dovich v_c(R_c)
- Collapse ellissoidale come troncamento della LPT (Monaco 1997)

**Fragmentation:**
- Particelle ordinate per z_c decrescente
- Vicini lagrangiani = 6 nearest neighbors sulla griglia iniziale
- Seed halo: massimi locali di z_c
- Accretion: d ≤ f_a * R_M (R_M = M^{1/3} in unità grid spacing)
- Merging: d ≤ f_m * R_M tra aloni candidati
- Filaments: particelle che non accretano
- Parametri calibrati: f_a=0.18, f_m=0.35, f_r=0.7
- Dipendenza temporale: una particella può accretarsi solo su aloni già formati
  → questa dipendenza è il problema centrale della parallelizzazione

**Dipendenza dati critica:**
I vicini lagrangiani sono per costruzione locali alla griglia iniziale.
Con decomposizione dominio MPI, le dipendenze cross-rank esistono solo ai boundary.
La domanda aperta: il loop interno al rank può diventare OpenMP?
E/o esistono metodi approssimati dalla letteratura che scalano meglio?

---

## Preprocessor options chiave (Makefile)

| Flag | Effetto |
|------|---------|
| `-DTWO_LPT` | Abilita displacements 2LPT |
| `-DTHREE_LPT` | Abilita displacements 3LPT (se entrambi, usa 3LPT) |
| `-DPLC` | Past Light Cone reconstruction |
| `-DELL_CLASSIC` | Dinamica collasso ellissoidale classica |
| `-DCLASSIC_FRAGMENTATION` | Usa algoritmo di frammentazione classico |
| `-DSNAPSHOT` | Output snapshot di particelle |
| `-DLONGIDS` | IDs a 64 bit |
| `-DUSE_FFT_THREADS` | Threading OpenMP per FFTW (richiede OMP=YES) |
| `-DSCALE_DEPENDENT` | Cosmologie con growth rate scale-dependent |
| `-DMOD_GRAV_FR` | Gravità modificata f(R) |

---

## Convenzioni di codice

- Linguaggio: C (standard C11)
- Compilatore primario: clang
- Parallelismo CPU: MPI + OpenMP
- Parallelismo GPU: OpenMP 5.x target offload + CUDA (nvcc) dove necessario
- FFT: heFFTe (già integrato, non sostituire)
- Sorting: radix sort (già integrato, non sostituire con qsort)
- Documentazione: Doxygen
- Ogni funzione pubblica deve avere docstring Doxygen completa
