# Redesign Plan

> Documento dinamico. Aggiornare dopo ogni decisione architetturale significativa.
> Leggere sempre prima di proporre modifiche strutturali al codice.

---

## Obiettivo generale

Ristrutturare Pinocchio per eseguire efficientemente su architetture GPU
(CINECA Leonardo, A100/H100), mantenendo correttezza fisica verificata
dalla HMF. Il redesign è incrementale e basato su branch separati da mergere.

---

## Stato attuale dei branch

### `feature/parallel-radix-sort` ✅ In produzione
- Sostituisce il sort parallelo nella fragmentazione con radix sort
- Verificato e testato, risultati corretti
- Occupa solo 2.4% del tempo totale — non è più un bottleneck
- **Da mergere con `refactoring_leonardo`**

### `refactoring_leonardo` ✅ In produzione
- FFT migrata da FFTW3/PFFT a **heFFTe**
- heFFTe è GPU-portable: supporta cuFFT (NVIDIA), rocFFT (AMD), oneMKL (Intel)
- Decomposizione pencil (più scalabile di slab ad alto numero di rank MPI)
- In produzione su CINECA Leonardo con GPU A100/H100
- Differisce da `feature/parallel-radix-sort` principalmente nella parte
  collapse times — merge atteso pulito, nessun conflitto noto

### Merge target ⏳ Prossimo passo
- Unione di `feature/parallel-radix-sort` + `refactoring_leonardo`
- Branch base per tutto il redesign successivo
- Merge dovrebbe essere pulito (parti separate del codice)

---

## Roadmap redesign

```
FASE 1 — Merge branch [prossimo passo immediato]
  └── feature/parallel-radix-sort + refactoring_leonardo → branch unificato

FASE 2 — Redesign fragmentazione [obiettivo principale]
  ├── 2a. Analisi algoritmica del loop Groups (build_groups.c)
  ├── 2b. Scelta approccio di parallelizzazione (da definire)
  ├── 2c. Redesign strutture dati per GPU
  ├── 2d. Implementazione OpenMP offload
  └── 2e. Validazione HMF su runs di riferimento

FASE 3 — Ottimizzazione Redistribution [secondario]
  └── Riduzione comunicazione MPI inter-rank nella fragmentazione
```

---

## Scope

### IN scope
- Fragmentazione: parallelizzazione fine-grained + GPU offload (OpenMP offload)
- Merge dei due branch attivi
- Redesign strutture dati in `fragment.c`, `build_groups.c`, `fragment.h`
- Riduzione overhead MPI in `distribute.c` (Redistribution)

### OUT scope — non toccare
- Neutrini, gravità modificata, scale-dependent growth
- Past Light Cone (PLC)
- 2LPT / 3LPT (stabili, verificati, in produzione)
- heFFTe integration (già funzionante in `refactoring_leonardo`)
- Parametri fisici `f_a`, `f_m`, `f_r`

---

## Decisioni architetturali prese

| Data | Decisione | Motivazione |
|------|-----------|-------------|
| — | Adottare heFFTe come backend FFT | Portabilità GPU, decomposizione pencil |
| — | Adottare radix sort parallelo | Performance sort nella fragmentazione |
| — | OpenMP offload come modello GPU | Portabilità, evita vendor lock-in (CUDA) |
| — | Mantenere C (non C++) per il core | Compatibilità con codebase esistente |

---

## Approccio algoritmico fragmentazione — DA DEFINIRE

> Questa sezione va completata in sessione Claude Code dopo analisi
> diretta di `fragment.c` e `build_groups.c`.

### Problema core
Il loop Groups è sequenziale per costruzione: la decisione per la
particella `i` dipende dallo stato degli aloni costruiti dalle
particelle precedenti (quelle con `z_c` maggiore).
Le dipendenze sono però **locali nello spazio Lagrangiano** (raggio 1
grid spacing — solo 6 vicini).

### Opzioni identificate (da duscutere e valuatre sul codice reale)

**A — Wavefront / Level-set parallelism**
Partiziona particelle in bin di `z_c`. Dentro ogni bin: parallelismo
completo. Tra bin: dipendenze causali rispettate.
- Pro: concettualmente semplice, preserva fisica
- Contro: granularità da calibrare, overhead bin boundary
- Da verificare: distribuzione di `z_c` nel codice reale

**B — Union-Find parallelo**
Reformula accretion/merging come connected components su grafo.
Algoritmi paralleli: Shiloach-Vishkin, ECL-CC.
- Pro: scalabilità GPU nativa, ben studiato in letteratura
- Contro: deve rispettare causalità temporale di `z_c`
- Da verificare: se la causalità è incorporabile nel grafo

**C — Domain decomposition + halo stitching**
Ogni GPU processa un sottodominio Lagrangiano in seriale.
Stitching degli aloni ai bordi tra domini.
- Pro: approccio conservativo, preserva HMF con alta probabilità
- Contro: overhead stitching, complessità implementativa ai bordi
- Da verificare: dimensione tipica degli aloni vs sottodominio

**D - Apertura a nuove ipotesi di algoritmo da valutare**
Pensare ad altri algorimti.

### Prossimi passi per la scelta
1. Leggere `fragment.c` e `build_groups.c` in sessione Claude Code
2. Identificare la struttura dati centrale e il loop critico
3. Misurare la distribuzione delle dipendenze tra particelle vicine in `z_c`
4. Scegliere approccio e prototipare

---

## Invarianti di correttezza

Qualsiasi implementazione deve:
1. Riprodurre la HMF entro la varianza statistica del campione
   (non floating point — l'ordine delle operazioni cambierà)
2. Produrre merger trees consistenti (no loop, no aloni orfani)
3. Girare correttamente su CPU (fallback OpenMP host) prima di
   testare il target GPU
4. Superare i test in `tests/only_HMF_tests/` su tutte le configurazioni
   LCDM di riferimento

---

## Note e decisioni aperte

- Tolleranza numerica accettabile sulla HMF dopo GPU offload:
  **da concordare con il team prima dei test**
- Strategia di test incrementale: CPU seriale → CPU OpenMP → GPU offload
- Il branch di sviluppo per il redesign fragmentazione va creato
  dal merge target (Fase 1)
