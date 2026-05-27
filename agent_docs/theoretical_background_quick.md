# Theoretical Background — Quick Reference

> Livello operativo. Leggi questo file per qualsiasi task sul codice.
> Per dettagli fisici e matematici, leggi `theoretical_background_deep.md`
> solo se esplicitamente richiesto.

---

## Cosa fa Pinocchio

Pinocchio genera cataloghi di aloni di materia oscura senza simulazione N-body completa.
Usa Lagrangian Perturbation Theory (LPT) per predire quando e dove ogni elemento
di massa collassa, poi raggruppa i collassi in aloni tramite un algoritmo di
fragmentazione. Output: cataloghi di aloni con massa, posizione, velocità, merger tree.

---

## Pipeline di esecuzione (5 stadi)

```
1. INIZIALIZZAZIONE
   Legge parameter file, alloca griglia, genera/legge white noise

2. LPT  [src/LPT.c, src/fmax.c, src/fmax-fftw.c / fmax-heffte.c]
   Calcola spostamenti 2LPT o 3LPT via FFT
   Produce il campo di velocità e spostamento per ogni particella

3. COLLAPSE TIMES  [src/collapse_times.c, src/cosmo.c]
   Per ogni particella e ogni smoothing radius R:
   applica ellipsoidal collapse → ricava z_c(R)
   Registra z_c_max, R_c, velocità Zel'dovich v_c

4. FRAGMENTAZIONE  [src/fragment.c]  ← OGGETTO DEL REDESIGN
   Ordina particelle per z_c decrescente (ora: radix sort parallelo)
   Assegna ogni particella a: seed halo / accretion / merging / filamento
   Output: lista aloni con particelle member

5. OUTPUT  [src/write_halos.c, src/write_snapshot.c]
   Scrive cataloghi, mass function, merger histories
```

---

## Moduli sorgente — mappa rapida

| File | Responsabilità |
|------|---------------|
| `pinocchio.c` | Main, orchestrazione pipeline |
| `pinocchio.h` | Header globale, macro, strutture dati |
| `variables.c` | Variabili globali MPI, griglia, cosmologia |
| `allocations.c` | Gestione memoria (alloca/dealloca griglia) |
| `initialization.c` | Setup griglia, parametri cosmologici |
| `LPT.c` | Spostamenti LPT 2° e 3° ordine |
| `fmax.c` | Calcolo Fmax (peak del campo deformazione) |
| `fmax-fftw.c` | Backend FFT con FFTW3/PFFT |
| `fmax-heffte.c` | Backend FFT con heFFTe (GPU-portable) ← attivo in `refactoring_leonardo` |
| `collapse_times.c` | Ellipsoidal collapse, z_c per ogni particella |
| `cosmo.c` | Funzioni cosmologiche (H(z), D(z), spline) |
| `def_splines.h` | Spline cosmologiche precompilate |
| `fragment.c` | Fragmentazione: sort → seed → accretion → merge |
| `fragment.h` | Strutture dati fragmentazione (halo, gruppi) |
| `build_groups.c` | Costruzione gruppi/aloni durante fragmentazione |
| `distribute.c` | Distribuzione MPI delle particelle |
| `GenIC.c` | Generazione condizioni iniziali (white noise) |
| `ReadWhiteNoise.c` | Lettura white noise esterno |
| `ReadParamfile.c` | Parsing parameter file |
| `Pk_from_CAMB.c` | Lettura power spectrum da CAMB |
| `run_planner.c` | Pianificazione run, stima memoria |
| `write_halos.c` | Output cataloghi aloni |
| `write_snapshot.c` | Output snapshot |

---

## Branch attivi e loro scopo

| Branch | Stato | Contenuto |
|--------|-------|-----------|
| `feature/parallel-radix-sort` | ✅ Produzione | Radix sort parallelo per ordinamento particelle in fragmentazione |
| `refactoring_leonardo` | ✅ Produzione | FFT migrata a heFFTe, portabile su GPU (A100/H100 CINECA) |
| *(merge target)* | 🔧 Obiettivo | Unione dei due branch → base per redesign fragmentazione |

---

## Scope del redesign corrente

**IN scope:**
- Fragmentazione: parallelizzazione fine-grained + GPU offload (OpenMP offload)
- Merge `feature/parallel-radix-sort` + `refactoring_leonardo`
- Architettura: separazione logica sort / accretion / merging / output

**OUT scope (non toccare):**
- Neutrini, gravità modificata, scale-dependent growth
- Past Light Cone (PLC)
- 2LPT / 3LPT (stabili, in produzione)
- heFFTe integration (già funzionante)

---

## Parametri fisici critici (non modificare senza consenso esplicito)

| Parametro | Valore | Ruolo |
|-----------|--------|-------|
| `f_a` | 0.18 | Linking length accretion |
| `f_m` | 0.35 | Linking length merging |
| `f_r` | 0.7 | Correzione risoluzione griglia |

Questi parametri sono calibrati su simulazioni N-body. Cambiarli invalida
la mass function. Qualsiasi modifica alla fragmentazione deve produrre
**risultati numericamente identici** a parità di parametri.

---

## Target HPC

- **Cluster**: CINECA (Leonardo)
- **GPU**: A100 / H100
- **Modello GPU**: OpenMP offload (target direttiva `#pragma omp target`)
- **MPI**: OpenMPI
- **FFT**: heFFTe (già integrato)
- **Compilatore**: clang (preferito)

---

## Invariante di correttezza

Il test di riferimento è la **Halo Mass Function (HMF)**.
Qualsiasi modifica alla fragmentazione deve riprodurre la HMF dei run
di riferimento in `tests/` entro la precisione numerica del floating point.
I plot di validazione sono in `HMF_Validation/`.
