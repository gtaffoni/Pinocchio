# PINOCCHIO — Branch status

Stato dei branch di sviluppo e dei rami archiviati. Aggiornato 2026-06.

## Base di sviluppo attiva

### `feature/leonardo-radix` ✅ BASE STABILE
Versione di lavoro di Tiago (Leonardo / heFFTe, presa da `src_Leonardo/`) con
**l'unica modifica** del radix sort parallelo al posto di `qsort` nella
frammentazione.

- `build_groups.c` invariato rispetto alla versione di Tiago (nessun path
  `build_groups_parallel`).
- Radix sort: doppia passata stabile (frag_pos asc, poi Fmax desc) che riproduce
  **esattamente** `index_compare_F` (tie-break per posizione lagrangiana, richiesto
  da `RECOMPUTE_DISPLACEMENTS`); sort per posizione via radix ascendente.
- **Validato bit-identico a qsort** su cataloghi/mass-function/histories/PLC a
  np = 1, 2, 4 (MPI) e OMP = 1, 2, 8 — box 128³, output z = 2, 1, 0.5, 0, con
  PLC + SNAPSHOT + RECOMPUTE_DISPLACEMENTS + ALLTOALL attivi. Sort ~2.6× più veloce.

**Tutto il nuovo sviluppo (FastFrag, parallelizzazione a colori/tiling, GPU
offload, ecc.) riparte da questo branch.** Tenere un worktree fisso su
`feature/leonardo-radix` per i confronti di regressione delle implementazioni future.

## Rami archiviati (NON usare — strade morte)

Archiviati come tag annotati `archive/<nome>` (la storia è preservata; i branch
sono stati rimossi). Decisione condivisa col team, 2026-06.

| Tag archivio | Cos'era | Perché è morto |
|---|---|---|
| `archive/parallel-radix-sort` | radix sort + altro | Nasconde una parallelizzazione OpenMP non documentata di `build_groups` (`build_groups_parallel`) con bug bloccanti; non compila stand-alone (manca `gpu_offload.h`); forcato da un master vecchio (regredisce ~12 fix). **Il radix sort, corretto, è stato estratto in `feature/leonardo-radix`.** |
| `archive/omp_parallel_groups` | tiling OpenMP 8-colori della frammentazione | Origine di `build_groups_parallel`: la Fase 2 accresce le particelle di bordo con le masse **finali** dei gruppi → violazione di causalità; i cataloghi **dipendono da `OMP_NUM_THREADS`** (riproducibilità rotta). |
| `archive/fastfrag-subvolume` | FastFrag 8-pass su sub-volumi | Costruito sulla base aggrovigliata di `parallel-radix-sort`. Audit: validazione stantia; posizioni degli aloni mai ritradotte dal frame del tile a quello della subbox; `aux2` non è la distanza minima dal bordo (aloni troncati dichiarati safe → bias HMF); problemi `FRAGFIELDS`/`Nalloc`. |

### Recuperare un ramo archiviato
```sh
git branch feature/<nome> archive/<nome>      # ricrea il branch dal tag
# es: git branch feature/omp_parallel_groups archive/omp_parallel_groups
```

## Note
- Le **idee** (subvolume tiling, parallelizzazione a colori) non sono morte: vanno
  re-implementate pulite su `feature/leonardo-radix`, con validazione HMF
  bit-identica vs la baseline qsort come gate. La via per l'algoritmo va concordata
  con l'auditor (cosmologo/HPC) prima dell'implementazione.
- Per il dettaglio degli audit vedere la cronologia della sessione di sviluppo.
