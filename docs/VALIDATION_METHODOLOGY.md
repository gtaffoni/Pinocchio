# Metodologia di validazione di regressione per PINOCCHIO

**Versione:** 1.0 — 2026-06-12
**Base di riferimento:** branch `feature/leonardo-radix` (heFFTe + radix sort), validato bit-identico rispetto a qsort su cataloghi, mass function, histories e PLC a np = 1/2/4 e OMP = 1/2/8.
**Scopo:** dare  uno strumento indipendente, self-contained (Python + numpy), eseguibile su qualunque server (incluso il server GPU), che dopo ogni sviluppo (porting GPU, FastFrag, tiling a colori, ottimizzazioni) risponda alla domanda: *il nuovo codice produce gli stessi gruppi, con lo stesso numero di particelle, nelle stesse posizioni — o la differenza fisica è troppo grande?*

**Principio fondante (non negoziabile):** contare il numero di gruppi o di picchi NON basta. L'audit di FastFrag ha mostrato un caso reale in cui il numero di aloni e la mass function erano compatibili con il riferimento, ma le posizioni finali erano *tutte* sbagliate (output nel frame del sottovolume invece che nel frame globale). Un test basato solo su conteggi e HMF avrebbe dato PASS a un codice rotto. La validazione deve quindi essere **alone-per-alone**, sfruttando l'identificativo lagrangiano.

---

## Il perno della metodologia: l'ID lagrangiano (`name`)

Ogni alone PINOCCHIO porta con sé il campo `name`: l'indice di griglia della posizione lagrangiana del picco di collasso che ha generato il gruppo. Questo ID:

- è un intero (`int64` nel catalogo, `uint64` in histories/PLC), deterministico, indipendente dall'ordinamento di scrittura, dal numero di task che scrivono e dall'ordine delle operazioni parallele;
- identifica *lo stesso oggetto fisico* in due run con le stesse condizioni iniziali (stesso seed, stessa griglia);
- permette il **matching uno-a-uno** tra catalogo di riferimento e catalogo candidato senza alcuna euristica spaziale.

Tutte le metriche di livello 1 e 2 sono costruite su questo matching.

### Formati di output (verificati su ReadPinocchio5.py e su output reali)

Output prodotti da un run (per ogni redshift `<z>` della OutputList e per run `<run>`):

| File | Contenuto |
|---|---|
| `pinocchio.<z>.<run>.catalog.out` | catalogo aloni a z fissato |
| `pinocchio.<z>.<run>.mf.out` | mass function binnata (ASCII sempre) |
| `pinocchio.<run>.histories.out` | merger histories (alberi) |
| `pinocchio.<run>.plc.out` | catalogo sul cono luce (se PLC attivo) |
| `pinocchio.<run>.nz.out` | n(z) sul cono luce (se PLC attivo) |

**Catalogo binario** (formato nuovo, record length 56, letto da `ReadPinocchio5.catalog`):

```python
cat_dtype = [('name',  np.int64),       # ID lagrangiano del picco  <-- chiave di matching
             ('Mass',  np.float32),     # massa [Msun/h]
             ('pos',   np.float32, 3),  # posizione finale [Mpc/h]
             ('vel',   np.float32, 3),  # velocità [km/s]
             ('posin', np.float32, 3),  # posizione iniziale (lagrangiana) [Mpc/h]
             ('npart', np.int32)]       # numero di particelle
```

**Catalogo ASCII** (`CatalogInAscii`): colonne `group ID`, `group mass (Msun/h)`, `initial position (Mpc/h)` (3), `final position (Mpc/h)` (3), `velocity (km/s)` (3), `number of particles`.

> **ATTENZIONE — quantizzazione ASCII:** nell'ASCII le posizioni sono stampate con 2 decimali (passo 0.01 Mpc/h) e le velocità idem. Per la validazione di Livello 1 e per le code fini delle distribuzioni di Livello 2 va usato l'**output binario** (`CatalogInAscii` disattivato). L'ASCII è accettabile solo come smoke test o quando la soglia su |Δx| è ≫ 0.01 Mpc/h.

**mf.out** (ASCII, già pronto per il test): colonne 1) massa del bin [Msun/h], 2) n(m) [Mpc⁻³ Msun⁻¹ h⁴], 3) limite superiore +1σ, 4) limite inferiore −1σ, 5) **numero di aloni nel bin**, 6) n(m) analitica (Watson et al. 2013). Le colonne 3–4 forniscono direttamente le barre Poisson per bin: il test le riusa, non le ricalcola.

**histories** (letto da `ReadPinocchio5.histories`): per ogni branch `name` (uint64, ID radice/branch), `nickname` (indice nel tree), `link`, `merged_with`, `mass_at_merger`, `mass_of_main`, `z_merging`, `z_peak`, `z_appear`; più `Ntrees`, `Nbranches` (array per tree), `pointers`.

**plc** (letto da `ReadPinocchio5.plc`): `name`, `truez`, `pos`(3), `vel`(3), `Mass`, `theta`, `phi`, `vlos`, `obsz`.

### Vincoli stabiliti sul confronto

1. **Confrontare sempre a PARI numero di rank MPI** (e idealmente pari OMP, anche se OMP è stato verificato ininfluente sul leonardo-radix). A np > 1 lo strato di bordo della decomposizione introduce un'approssimazione nota e legittima: run a np diversi NON sono identici e non devono esserlo.
2. **np = 1 è il ground truth deterministico**: nessun bordo, nessuna comunicazione. È la configurazione primaria del golden reference.
3. **L'integrità del run viene verificata PRIMA di qualunque confronto.** Confrontare file di output può ingannare se il run candidato è abortito lasciando file stantii di un run precedente: è già successo. Il test fallisce con un codice di uscita dedicato (exit 2, distinto dal FAIL fisico) se mancano file, se i file sono incoerenti o se il run non risulta completato.

---

## A. Metodologia a tre livelli

Ogni modifica al codice dichiara **a priori, prima dell'implementazione**, il livello di equivalenza atteso. Il test verifica quel livello. Il declassamento (es. da L0 a L2) è una **decisione esplicita e motivata** — registrata nel log del test e nel PR — mai un ripiego perché "il diff non passa". Se un cambiamento dichiarato L0 non passa L0, il primo sospetto è un bug, non una tolleranza troppo stretta.

### Livello 0 — Bit-identico

- **Verifica:** `cmp` byte-per-byte di tutti i file di output (cataloghi binari, mf.out, histories, plc, nz) tra riferimento e candidato.
- **Cosa garantisce:** equivalenza totale: stessa matematica, stesso ordine di operazioni FP, stesso ordine di scrittura. Nessuna possibilità di regressione fisica.
- **Quando si applica:** modifiche algebricamente equivalenti su CPU, stesso compilatore, stesse flag, stesso np/OMP. Esempio dimostrato: la sostituzione qsort → radix sort è risultata bit-identica a np = 1/2/4, OMP = 1/2/8. Refactoring, riorganizzazione di memoria, cambi di I/O bufferizzato che non toccano i valori, ottimizzazioni che non riordinano somme FP.
- **Nota:** L0 a pari np vale anche a np > 1 — la decomposizione è la stessa, quindi anche lo strato di bordo è identico.

### Livello 1 — Identico a meno di riordino

- **Verifica:** dopo sort per `name` di entrambi i cataloghi: (a) gli insiemi di `name` coincidono esattamente; (b) `npart` identico per ogni alone; (c) i campi float (`Mass`, `pos`, `vel`, `posin`) identici bit a bit o entro 1–2 ulp; (d) histories: stessi alberi, stessi branch, stessi campi.
- **Cosa garantisce:** la matematica è identica, è cambiato solo l'ordine in cui i risultati vengono prodotti o scritti (es. diverso ordinamento parallelo, diverso ordine dei task in scrittura, diversa suddivisione dei blocchi). Nessuna differenza fisica, nemmeno minima.
- **Quando si applica:** cambi della strategia di sort/scrittura, riorganizzazione dei task di output, NumFiles diverso, scheduling che cambia l'ordine dei record ma non i valori.
- **Richiede output binario** (l'ASCII a 2 decimali maschererebbe differenze reali sotto 0.01 Mpc/h — e nasconderebbe quindi un declassamento di fatto a L2).

### Livello 2 — Equivalente fisico entro tolleranza

- **Verifica:** matching per `name`, distribuzioni delle differenze, mass function entro errori statistici, coerenza delle histories — con soglie numeriche (sezione C).
- **Cosa garantisce:** il codice candidato produce *gli stessi oggetti fisici* — stessi gruppi (matching ≈ totale), stesse masse (ΔNpart concentrato su 0), stesse posizioni (|Δx| una piccola frazione della cella) — con differenze compatibili con il riordino floating-point e/o con l'approssimazione dichiarata. NON garantisce identità: garantisce che la differenza è fisicamente irrilevante e quantificata.
- **Quando si applica:**
  - **GPU:** riduzioni con ordine diverso, FMA, libm diversa, precisione mista → non ci si può aspettare bit-identicità nemmeno a np = 1. È il caso d'uso principale.
  - **Approssimazioni algoritmiche dichiarate:** FastFrag, tiling a colori, modifiche allo strato di bordo. Qui la differenza è *strutturale*, non solo FP: servono soglie dedicate (profilo separato in `thresholds.json`).
  - Cambi di compilatore/flag aggressive (`-Ofast`, ecc.) sul codice CPU.

### Regola di ingaggio

| Modifica | Livello dichiarato |
|---|---|
| Refactoring, sort, I/O, memoria (CPU, stesse flag) | **L0** |
| Cambio ordine di scrittura/sort parallelo | **L1** (L0 se possibile) |
| Porting GPU di un kernel | **L2**, profilo `gpu-fp` |
| FastFrag, tiling, modifiche al bordo | **L2**, profilo `algo-approx` |
| Cambio compilatore / flag FP | **L2**, profilo `gpu-fp` |

Se un cambiamento dichiarato L2 passa L1 o L0: ottimo, registrarlo. Il contrario richiede indagine e decisione esplicita.

---

## B. Metriche

Ordine di esecuzione: prima i controlli di integrità (B.0), poi il matching (B.1) che alimenta tutte le metriche per-alone (B.2), poi le metriche statistiche (B.3–B.5). I conteggi globali (B.6) sono solo una sanity necessaria-non-sufficiente.

### B.0 Integrità del run (gate, non metrica)

Per ciascuna directory (riferimento e candidato):

1. **Completezza:** per ogni z nella OutputList devono esistere `catalog.out` e `mf.out`; più `histories.out`; più `plc.out`/`nz.out` se il PLC è attivo nel manifest. File mancante → exit 2.
2. **Leggibilità e coerenza interna:** `ReadPinocchio5` verifica già la consistenza dimensionale dei file binari (record length, file size attesa vs reale) e fallisce su file troncati. Un fallimento di lettura → exit 2.
3. **Coerenza temporale:** tutti i file di output devono essere più recenti dell'inizio del run (mtime ≥ timestamp di lancio registrato, se disponibile) e mutuamente coerenti (il catalogo a z = 0, scritto per ultimo, non può precedere quello a z = 2). Questo intercetta il caso reale dei *file stantii di un run abortito*.
4. **Coerenza di setup:** il manifest del golden (sezione D) registra seed, BoxSize, GridSize, OutputList, MinHaloMass, np. Il candidato deve dichiarare gli stessi valori (letti dal parameter file nella directory candidata): mismatch → exit 2 con messaggio esplicito. Confrontare run con seed o griglia diversi non ha senso e non deve poter accadere silenziosamente.
5. **np uguale:** ref e candidato devono essere stati eseguiti con lo stesso numero di task MPI (dal manifest / dal log). Mismatch → exit 2.

### B.1 Matching alone-per-alone via `name` (per ogni redshift)

```
matched   = intersezione dei name        (np.intersect1d, return_indices=True)
only_ref  = name presenti solo nel riferimento
only_cand = name presenti solo nel candidato
```

Statistiche sintetiche:

- **f_match** = |matched| / |ref| — globale e **per bin di npart**: [10–19], [20–99], [≥100]. Gli aloni alla soglia `MinHaloMass` (10 particelle) sono i più fragili: un alone da 10 particelle che ne perde una *esce dal catalogo* — è il canale dominante di non-match legittimo. Gli aloni grandi invece non hanno scuse: un non-match a npart ≥ 100 è quasi certamente un bug.
- **f_mass_unmatched** = massa totale degli only_ref / massa totale del riferimento (idem per only_cand). Pesa i non-match per importanza fisica.
- Lista esplicita (nel report) dei non-match con npart ≥ 100, con posizione e massa: vanno ispezionati uno per uno.

### B.2 Metriche sui match (per ogni redshift)

Per ogni coppia matched (stesso `name`):

- **ΔNpart** = npart_cand − npart_ref. Statistiche: frazione con ΔNpart = 0; p99 di |ΔNpart|/npart_ref; max |ΔNpart| con npart ≥ 100. ΔNpart è la metrica più "dura": intero, niente rumore FP — misura direttamente se l'algoritmo di accrescimento/merging ha preso le stesse decisioni. (`Mass` è ridondante con npart × massa particella; si controlla la proporzionalità come consistenza interna.)
- **|Δx|** = distanza tra `pos` con **convenzione di immagine minima** (il box è periodico: un alone a x = 0.01 e x = 299.99 in un box da 300 dista 0.02, non 299.98). Riportata in **unità di cella** Δg = BoxSize/GridSize e in Mpc/h. Statistiche: mediana, p99, max; scatter plot opzionale |Δx| vs npart.
- **|Δv|** in km/s, e relativa |Δv|/|v|. Stesse statistiche. Le velocità sono derivate LPT degli stessi spostamenti: se le posizioni sono a posto e le velocità no, il bug è nel calcolo delle velocità (o nei fattori di crescita).
- **Δposin** (check di frame): `posin` è la posizione lagrangiana iniziale dell'alone, una quantità *simile a un centro di massa* sulle posizioni iniziali delle particelle membro. A Livello 0/1 deve essere esatta. A Livello 2 (fisico/GPU) porta rumore floating-point: stesso codice+config ma compilatore diverso (es. **nvc vs gcc**: contrazione FMA, intrinseche math) o aritmetica GPU la spostano di pochi ulp float32 (**~1e-5 celle** misurate a Grid=64). Si misura quindi in **frazione di cella** con soglia `posin_max_cells` (default gpu-fp 1e-3): un bug di frame/indicizzazione sposta `posin` di **celle intere** (≥ 1, ~1e5 ulp), ordini di grandezza oltre la soglia. È un check quasi gratuito e molto potente contro errori di frame/indice (il bug FastFrag sarebbe stato preso anche da qui). Nota: il rumore FP di `posin` cresce lievemente con GridSize e con la dimensione degli aloni → ricalibrare per griglie di produzione grandi.

### B.3 Mass function bin-per-bin (per ogni redshift)

Da `mf.out`, usando il **numero di aloni per bin** (colonna 5) e le barre ±1σ (colonne 3–4):

- residuo per bin in unità di σ: r_i = (n_cand,i − n_ref,i) / σ_i, con σ_i dalla semi-ampiezza delle barre del riferimento;
- statistiche: max |r_i| sui bin con ≥ 50 aloni; **sign test** (frazione di bin con r_i > 0 — un eccesso sistematico di segno indica un bias, anche se ogni singolo bin è "dentro 1σ");
- KS a due campioni sulle distribuzioni di massa dei cataloghi (complementare al binning).

**Avvertenza fondamentale:** i due run condividono le stesse condizioni iniziali, quindi NON sono realizzazioni indipendenti: la fluttuazione Poisson è un **tetto massimo generoso**, non il valore atteso. Per un porting GPU corretto i residui devono essere *molto* sotto 1σ (tipicamente |r| < 0.1 con f_match ~ 1). Un candidato che sta "a 1σ su tutti i bin" rispetto a un run con le stesse IC è sospetto, non conforme. Per questo la HMF da sola non basta: è la metrica meno sensibile della suite.

### B.4 Merger histories

Matching dei tree per `name` della radice (primo branch del tree), poi branch-per-branch per `name`:

- frazione di tree matched; frazione di tree matched con stesso `Nbranches`;
- per i branch matched: identità di `merged_with` (riferito via `name` del partner, non via indice `nickname`, che dipende dall'ordinamento interno); **merger-flip rate** = frazione di branch in cui cambia il partner di merger o l'ordine main/satellite; |Δz_merging|, |Δz_peak|, |Δz_appear| (p99);
- coerenza `Ntrees`/`Nbranches_tot` globali come sanity.

Le histories sono sensibili alla *sequenza temporale* delle decisioni di merging: errori che i cataloghi a z fissato compensano (es. un merger anticipato di uno step) qui emergono.

### B.5 Past light cone (se attivo)

- Conteggio totale e n(z) (da `nz.out`) bin-per-bin con criterio Poisson come B.3;
- matching per chiave (`name`, round(truez, 4)) — lo stesso alone può attraversare il cono a più epoche, la coppia disambigua; per i match: |Δtruez|, |Δtheta|, |Δphi|, |ΔMass|;
- il PLC interseca le traiettorie tra output: è la metrica più sensibile a differenze di timing, ma anche la più rumorosa ai bordi del cono (aloni che entrano/escono per spostamenti infinitesimi). **[Decisione Morgan]** se includere il PLC nel verdetto PASS/FAIL di L2 o tenerlo come diagnostica (raccomandazione: diagnostica nel primo ciclo di calibrazione, poi promozione a metrica con soglie proprie).

### B.6 Conteggi globali — sanity necessaria, MAI sufficiente

N_halos per z, N_trees, N_branches, N_plc: si riportano sempre, e una discrepanza grossolana (> qualche %) è un FAIL immediato senza bisogno di analisi fine. Ma la loro concordanza non dimostra nulla: **caso reale FastFrag** — numero di aloni e HMF compatibili, `pos` tutte nel frame sbagliato (offset del sottovolume non applicato). Solo il matching per `name` con confronto delle posizioni (B.2) e/o il check su `posin` lo intercetta. Qualunque proposta futura di "validare con i conteggi e la mass function" va respinta citando questo precedente.

---

## C. Soglie pass/fail

### C.1 Filosofia

- **Cambiamenti che DEVONO essere esatti** (refactoring CPU, stesso compilatore): la soglia è **zero**. L0: zero byte di differenza. L1: zero differenze su interi e insiemi, ≤ 1–2 ulp sui float (ulp misurate sul float32 scritto, cioè spacing di numpy a quel valore). Non esiste "quasi uguale" per un refactoring: o è equivalente o c'è un bug.
- **Cambiamenti approssimati/GPU:** soglie numeriche concrete, sotto riportate come **punto di partenza ragionato, da calibrare** (C.3). Due profili distinti in `thresholds.json`:
  - `gpu-fp` — solo riordino FP: soglie strette;
  - `algo-approx` — approssimazione algoritmica dichiarata (FastFrag, tiling): soglie da negoziare caso per caso, comunque non oltre la banda MPI intrinseca (C.4).

### C.2 Soglie L2 di partenza, profilo `gpu-fp` (a pari np, output binario)

| # | Metrica | Soglia iniziale | Razionale fisico |
|---|---|---|---|
| 1 | f_match, npart ≥ 20 | **≥ 99.9 %** | sopra 2× la soglia minima un alone non può sparire per il flip di una particella; solo eventi rarissimi di merging borderline sono tollerabili |
| 2 | f_match, tutti (npart ≥ 10) | **≥ 99 %** | alla soglia MinHaloMass il flip di una particella elimina/crea l'alone: ~1 % di churn è fisiologico per riordino FP |
| 3 | f_mass_unmatched | **≤ 0.1 %** | i non-match devono essere tutti aloni minuscoli; massa persa misurabile = oggetti grandi coinvolti = bug |
| 4 | non-match con npart ≥ 100 | **= 0** | un alone grande non matchato non ha spiegazione FP; ispezione obbligatoria |
| 5 | frazione ΔNpart = 0 | **≥ 95 %** | le decisioni di accrescimento sono discrete: il riordino FP può flippare solo particelle esattamente al bordo del criterio |
| 6 | p99 di \|ΔNpart\|/npart | **≤ 2 %** | una particella su un alone da 50; aloni grandi devono avere ΔN relativo ancora minore |
| 7 | mediana \|Δx\| | **≤ 0.01 Δg** | per gli aloni non toccati da flip, pos cambia solo per rumore FP nella media delle particelle: ≪ cella (Δg = 4.69 Mpc/h per box 300/grid 64; 3.91 per 500/128) |
| 8 | p99 \|Δx\| | **≤ 0.1 Δg** | la coda è dominata da aloni che hanno scambiato 1 particella: lo shift del centro di massa resta ≪ 1 cella |
| 9 | p99 \|Δv\| | **≤ 10 km/s** e ≤ 2 % relativo | stesso argomento di Δx trasferito alle derivate LPT; le v tipiche sono O(100–800) km/s |
| 10 | HMF | max \|r_i\| **≤ 0.2 σ_Poisson** (bin ≥ 50 aloni) e sign test con frazione positiva in [0.2, 0.8] | stesse IC ⇒ Poisson è tetto, non target (B.3); 0.2σ è già generoso quando f_match ≈ 1 |
| 11 | merger-flip rate | **≤ 0.1 %** dei branch | i merger sono decisioni discrete su coppie; solo coppie al limite del criterio di vicinanza possono flippare |
| 12 | tree matched con stesso Nbranches | **≥ 99.5 %** | segue da 2 e 11 |
| 13 | Δposin sui match | **= 0** a L0/L1; **≤ `posin_max_cells`** (gpu-fp 1e-3 celle) a L2 | centro-di-massa lagrangiano: porta rumore FP cross-compilatore/GPU (~1e-5 celle); un bug di frame lo sposta di celle intere (≥1), ben oltre la soglia |

Per il profilo `algo-approx` i valori delle righe 1–2, 5–8, 11 vanno rilassati **esplicitamente e singolarmente** in un file di soglie dedicato, con il vincolo C.4. **[Decisione Morgan]**: i valori `algo-approx` si fissano solo dopo aver visto la prima misura reale (es. FastFrag corretto) e la banda di calibrazione.

### C.3 Calibrazione (`--calibrate`)

I numeri in C.2 sono priors ragionati, non misure. Prima dell'uso in produzione:

1. **Rumore nullo:** `validate_run.py --level physical` tra due run *identici* del codice base (stesso binario, stesso np, due esecuzioni) — deve dare differenze esattamente nulle (il codice base è deterministico a pari configurazione). Verifica che il test stesso non introduca rumore.
2. **Banda intrinseca MPI:** confronto np = 1 vs np = 2 del codice base `leonardo-radix` (stesso seed). Questa è la **variazione fisicamente accettata** del metodo dovuta allo strato di bordo: il progetto convive già con essa. Output: gli stessi indicatori 1–12 misurati su questa coppia → archiviati come `calibration_band.json`.
3. **Regola di coerenza:** per il profilo `gpu-fp`, ogni soglia deve risultare **ben al di sotto** della banda np1-vs-np2 (un cambio FP non può legittimamente spostare i risultati più di quanto li sposti la decomposizione MPI accettata da sempre). Per `algo-approx`, la banda è il **limite superiore di negoziazione**: un'approssimazione che differisce dal riferimento più di quanto differiscano tra loro due decomposizioni MPI del codice base sta cambiando la fisica, non approssimando.
4. **[Decisione Morgan]** dopo la calibrazione: ratifica dei valori definitivi in `thresholds.json` (entrambi i profili), decisione sul PLC (B.5), eventuale soglia merger-flip diversa.

### C.4 Trattamento della decomposizione MPI

- Confronti **solo a pari np** (gate B.0.5).
- Golden primario a **np = 1** (ground truth, nessun bordo). Golden secondario a **np = 4** per validare anche il path di comunicazione/bordo del candidato a np = 4.
- La coppia (np = 1, np = 2) del codice base definisce la banda di variazione intrinseca (C.3.2) e dà significato fisico alle soglie.
- Mai usare un confronto np_A vs np_B (A ≠ B) come test di regressione: misurerebbe la decomposizione, non la modifica.

---

## D. Design del test eseguibile

### D.1 Interfaccia

```bash
python validate_run.py --reference <dir_o_golden> --candidate <dir> \
       [--level exact|reorder|physical]      # default: physical
       [--thresholds thresholds.json]        # default: thresholds incorporati (C.2, gpu-fp)
       [--profile gpu-fp|algo-approx]        # seleziona il profilo dentro thresholds.json
       [--calibrate]                         # modalità calibrazione: misura, non giudica
       [--make-golden <outdir>]              # congela il riferimento in un golden
       [--plots <dir>]                       # plot diagnostici opzionali (richiede matplotlib)
       [--report <file.txt|json>]            # report scritto su file oltre che a stdout
```

- `--level exact` = L0 (cmp byte-per-byte); `reorder` = L1; `physical` = L2.
- **Exit code: 0 = PASS, 1 = FAIL (fisico), 2 = run incompleto / errore di integrità / setup incoerente.** La distinzione 1 vs 2 è essenziale negli script CI e per non scambiare un run abortito per una regressione fisica (o viceversa).
- Dipendenze: Python ≥ 3.8, numpy. `matplotlib` solo se `--plots`. `ReadPinocchio5.py` distribuito nella stessa directory. Nessun Docker, nessuna infrastruttura.

### D.2 Golden reference congelato

`--make-golden` copia gli output del run di riferimento e scrive `manifest.json`:

```json
{
  "created": "2026-06-12T10:00:00",
  "code": {"branch": "feature/leonardo-radix", "commit": "<sha>", "compiler": "gcc-12 -O3", "flags": "TWO_LPT THREE_LPT ..."},
  "run":  {"np": 1, "omp": 1, "RunFlag": "valtest", "RandomSeed": 486604,
            "BoxSize": 300.0, "BoxInH100": true, "GridSize": 64,
            "MinHaloMass": 10, "outputs": [2.0, 1.0, 0.5, 0.0],
            "PLC": false, "CatalogInAscii": false, "NumFiles": 1},
  "files": {"pinocchio.0.0000.valtest.catalog.out": {"sha256": "...", "bytes": 123456}, "...": {}}
}
```

Il test consuma indifferentemente una directory di run "viva" o un golden: se trova `manifest.json` verifica gli SHA256 (il golden non può essere corrotto/sovrascritto silenziosamente) e usa i metadati per i gate di coerenza B.0.4–B.0.5. Il golden per il setup 64³ pesa pochi MB: si versiona o si trasferisce via scp senza problemi.

### D.3 Struttura del codice (scheletro implementabile)

```python
#!/usr/bin/env python3
"""validate_run.py — regression validation for PINOCCHIO outputs."""
import sys, os, json, hashlib, argparse
import numpy as np
import ReadPinocchio5 as rp

# ---------- exit codes ----------
PASS, FAIL, BROKEN = 0, 1, 2

DEFAULT_THRESHOLDS = {   # profilo gpu-fp, sezione C.2
  "f_match_core":      0.999,   # npart >= core_npart
  "core_npart":        20,
  "f_match_all":       0.99,
  "f_mass_unmatched":  0.001,
  "max_unmatched_npart": 100,   # nessun non-match con npart >= questo
  "frac_dnpart_zero":  0.95,
  "p99_dnpart_rel":    0.02,
  "median_dx_cells":   0.01,
  "p99_dx_cells":      0.1,
  "p99_dv_kms":        10.0,
  "hmf_max_sigma":     0.2,
  "hmf_sign_band":     [0.2, 0.8],
  "merger_flip_rate":  0.001,
  "tree_same_nbranches": 0.995 }

# ---------- discovery & integrity ----------
def discover_run(rundir):
    """Trova parameter file / manifest.json; ritorna dict con RunFlag, outputs(z),
    BoxSize, GridSize, MinHaloMass, np, PLC, e la lista dei file attesi."""

def integrity_check(run):
    """B.0: tutti i file attesi esistono; ReadPinocchio5 li legge senza errori;
    mtime coerenti (file z piu' bassi non piu' vecchi dei piu' alti);
    se manifest: verifica SHA256. Ritorna lista problemi; se non vuota -> BROKEN."""

def check_compatibility(ref, cand):
    """B.0.4-5: stesso seed, BoxSize, GridSize, outputs, MinHaloMass, np.
    Mismatch -> BROKEN con messaggio esplicito."""

# ---------- level 0 / level 1 ----------
def compare_exact(ref, cand):
    """L0: confronto byte-per-byte (hash sha256) di ogni file di output."""

def compare_reorder(ref, cand, ulp_tol=2):
    """L1: per ogni z carica i cataloghi, argsort per 'name';
    - set(name) identici (np.array_equal dopo sort), altrimenti FAIL
    - npart identici, altrimenti FAIL
    - float: diff <= ulp_tol * np.spacing(ref_val) elemento per elemento
    Idem histories (sort per (root_name, name)) e plc (sort per (name, truez))."""

# ---------- level 2 ----------
def match_by_name(cref, ccand):
    common, iref, icand = np.intersect1d(cref['name'], ccand['name'],
                                         return_indices=True, assume_unique=True)
    return iref, icand, np.setdiff1d(...), np.setdiff1d(...)

def periodic_delta(a, b, box):
    d = a - b
    return d - box * np.rint(d / box)     # immagine minima, per componente

def catalog_metrics(cref, ccand, box, grid):
    """B.1 + B.2: f_match (globale e per bin npart), f_mass_unmatched,
    lista non-match npart>=100; sui match: ΔNpart (frac zero, p99 rel),
    |Δx| in celle (mediana, p99, max), |Δv| (p99), Δposin in celle (frame check).
    Ritorna dict di numeri + tabelle per il report."""

def hmf_metrics(mf_ref_file, mf_cand_file):
    """B.3: legge colonne (m, n, up, low, nbin) con np.loadtxt;
    sigma = (up - low)/2; r_i sui bin con nbin>=50; max|r|, sign fraction."""

def histories_metrics(href, hcand):
    """B.4: tree root = name del primo branch di ogni tree (via pointers);
    match dei tree per root name; per i tree matched confronto Nbranches;
    branch matching per name dentro il tree; merged_with risolto in name
    del partner via nickname->name del tree; flip rate, |Δz_*| p99."""

def plc_metrics(pref, pcand):
    """B.5: chiave (name, round(truez,4)); n(z) da nz.out con criterio B.3."""

# ---------- verdict & report ----------
def verdict(metrics, thresholds):
    """Confronta ogni statistica con la soglia; ritorna PASS/FAIL e la lista
    [(metrica, valore, soglia, esito)] per il report. FAIL se una sola fallisce."""

def report(results, fmt='text'):
    """Tabella leggibile: per z, per famiglia di metriche; valori, soglie,
    PASS/FAIL marcato. In coda i conteggi globali (B.6) come info."""

def main():
    args = parse_args()
    ref, cand = discover_run(args.reference), discover_run(args.candidate)
    if (p := integrity_check(ref) + integrity_check(cand)
           + check_compatibility(ref, cand)):
        print_report(p); sys.exit(BROKEN)
    if args.level == 'exact':   ok = compare_exact(ref, cand)
    elif args.level == 'reorder': ok = compare_reorder(ref, cand)
    else:
        m = {z: catalog_metrics(...) for z in ref['outputs']}
        m['hmf'] = {z: hmf_metrics(...) for z in ref['outputs']}
        m['hist'] = histories_metrics(...)
        if ref['PLC']: m['plc'] = plc_metrics(...)
        if args.calibrate:
            json.dump(m, open('calibration_band.json','w')); sys.exit(PASS)
        ok = verdict(m, load_thresholds(args))
    sys.exit(PASS if ok else FAIL)
```

Note implementative:

- `assume_unique=True` in `intersect1d` è legittimo (i `name` sono univoci per costruzione) ma va *verificato* (`len(np.unique(name)) == len(name)`) come ulteriore integrity check: duplicati = bug grave del candidato.
- Il confronto in ulp (L1) usa `np.spacing` sul valore di riferimento, sul dtype effettivamente scritto (float32).
- La risoluzione di `merged_with` (B.4) deve passare per la mappa nickname→name del singolo tree, mai per indici globali: gli indici dipendono dall'ordinamento, i name no.
- `--plots`: istogrammi di |Δx| (log), ΔNpart, residui HMF con bande ±0.2/±1σ, scatter |Δx| vs npart. Diagnostica, mai parte del verdetto.

### D.4 Cosa NON fa lo script

Non compila né lancia PINOCCHIO (a differenza di `HMF_validation.py`): separazione netta tra esecuzione e giudizio. Il run lo lancia Morgan (o la CI) come preferisce; lo script giudica solo gli output. Questo lo rende portabile su qualunque server e immune da differenze di ambiente.

---

## E. Workflow GPU / cross-server

### E.1 Procedura passo-passo

**Sul server CPU (riferimento, una tantum per ogni setup):**

1. Build del branch `feature/leonardo-radix` (la base validata), flag standard di produzione, **`CatalogInAscii` disattivato** (output binario).
2. Run a np = 1, OMP = 1, con il parameter file di riferimento (seed 486604, OutputList z = 2, 1, 0.5, 0, MinHaloMass 10; setup minimo: GridSize 64 / BoxSize 300 come in `validation_runs/`; setup pieno: 128³ o superiore).
3. Run a np = 4 (stesso tutto): secondo golden, per validare il path multi-rank.
4. `python validate_run.py --reference run_np1 --make-golden golden_np1_<commit>` (idem np4). Il manifest congela commit, parametri, SHA256.
5. (Una tantum) Run a np = 2 e `--calibrate` contro il golden np = 1 → `calibration_band.json` (banda intrinseca MPI, sezione C.3).

**Trasferimento:** `tar czf golden.tgz golden_np1_* golden_np4_* calibration_band.json thresholds.json validate_run.py ReadPinocchio5.py` → scp sul server GPU. Pochi MB; il pacchetto contiene *tutto* il necessario, nessuna dipendenza dall'infrastruttura di origine.

**Sul server GPU:**

6. Build del branch GPU. Run candidate con lo **stesso parameter file estratto dal golden** (stesso seed, stessi output, stesso MinHaloMass, output binario), np = 1.
7. `python validate_run.py --reference golden_np1_<commit> --candidate run_gpu_np1 --level physical --profile gpu-fp --thresholds thresholds.json --plots diag/`
8. Se PASS a np = 1: ripetere a np = 4 contro `golden_np4`. La separazione np1/np4 distingue subito *errori del kernel GPU* (falliscono già a np = 1) da *errori di comunicazione/bordo* (compaiono solo a np = 4).
9. Archiviare il report accanto al commit candidato.

### E.2 Differenze ATTESE su GPU (non allarmanti se entro soglia)

- Float che differiscono di poche ulp ovunque (FMA, ordine delle riduzioni, libm device): è il motivo per cui il livello è L2 e non L0.
- Una frazione ≲ 1 % di aloni alla soglia di 10 particelle che appare/scompare (riga 2 di C.2): i tempi di collasso borderline flippano l'appartenenza di singole particelle.
- ΔNpart = ±1–2 su una piccola frazione di aloni; |Δx| ≪ cella, code corte; HMF indistinguibile (≪ 0.2σ).
- Qualche raro flip di merger tra coppie quasi simultanee.

### E.3 Campanelli d'allarme (FAIL da indagare, non da "ricalibrare")

- **f_match che crolla** (sotto ~99 % core): il candidato sta costruendo gruppi diversi — bug di fragmentation, non FP.
- **Δx con struttura sistematica**: stesso offset per molti aloni, o offset correlato con la posizione nel box ⇒ errore di frame/offset di sottovolume (esattamente il bug FastFrag); visibile immediatamente nell'istogramma per componente di Δx (non centrato su 0) e da Δposin ≠ 0.
- **Δposin oltre soglia (> `posin_max_cells`, default 1e-3 celle)**: bug di indicizzazione/frame (C.2 riga 13). Attenzione: un Δposin ≠ 0 ma di pochi ulp (~1e-5 celle) è solo rumore FP cross-compilatore/GPU, NON un bug — è il motivo per cui la soglia è sub-cella e non zero esatto.
- **Non-match con npart ≥ 100**: gli oggetti grandi sono robusti; perderne uno significa che una regione intera è trattata diversamente (bordo, tile, trasferimento dati incompleto).
- **HMF sistematicamente sopra/sotto** (sign test fuori banda) anche con residui piccoli: bias di soglia di collasso (es. precisione singola dove serviva doppia, costante fisica diversa, interpolazione GPU del tempo di collasso diversa da quella CPU).
- **ΔNpart grandi su aloni grandi** con posizioni corrette: criterio di accrescimento alterato (es. confronto `d <= R` vs `d < R` su GPU, o raggio calcolato in precisione diversa).
- **Velocità sbagliate con posizioni giuste**: fattori di crescita / derivate LPT calcolati diversamente sul device.
- **Differenze che crescono con z decrescente** (z = 0 molto peggio di z = 2): accumulo lungo la sequenza temporale della fragmentation ⇒ le decisioni divergono progressivamente; guardare le histories per individuare la prima epoca di divergenza.

---

## F. Limiti e onestà intellettuale

### F.1 Cosa questo test NON garantisce

- **Seed singolo:** la regressione è verificata su una (o poche) realizzazioni. Un bug che colpisce configurazioni rare (es. il picco esattamente sul bordo di un tile, vuoti estremi, il filamento più lungo del box) può non manifestarsi con il seed di riferimento. Il PASS dice "su questa realizzazione il codice riproduce il riferimento", non "il codice è corretto per ogni input".
- **Nessuna informazione sulla varianza cosmica o sull'accuratezza assoluta:** il test confronta PINOCCHIO con PINOCCHIO. Non dice nulla sulla bontà del metodo rispetto a N-body o all'universo reale: per quello esistono i confronti con le simulazioni (fuori scopo qui).
- **Scaling con il box non testato dal setup minimo:** 64³/128³ non esercita overflow di interi a 32 bit su griglie grandi (displacement, conteggi), né il load balancing reale, né NumFiles > 1, né Nslices > 1, né MaxMem stringenti. Un PASS a 64³ non copre questi percorsi di codice.
- **Copertura delle opzioni di compilazione:** il test valida la combinazione di flag del golden. Path non compilati (SCALE_DEPENDENT, RECOMPUTE_DISPLACEMENTS, snapshot, f(R), ...) restano non validati finché non hanno un golden dedicato.
- **Il test non sostituisce il giudizio:** un FAIL marginale su una metrica con tutte le altre verdi va capito, non aggirato alzando la soglia; un PASS con tutte le metriche esattamente al limite è sospetto (vedi B.3 sul "tutto a 1σ").

### F.2 Suite minima raccomandata prima della produzione GPU

**[Decisione Morgan]** sulla taglia massima sostenibile; raccomandazione di partenza:

| Cosa | Quanto | Perché |
|---|---|---|
| Seed | **3** (486604 + 2 nuovi) | esclude la sfortuna/fortuna della singola realizzazione; 3 è il minimo che distingue un caso patologico |
| Griglie/box | **64³/300** (rapido, nseg > 1), **128³/500** (setup storico di validazione), **256³/500** (cella più piccola ⇒ soglie Δx più severe in Mpc/h) | risoluzioni diverse esercitano profondità di ricorsione, dimensioni dei sottovolumi, bilanci di memoria diversi |
| np | **1 e 4** per ogni caso (confronti a pari np) | separa kernel da comunicazione |
| Output | PLC attivo, histories attive, binario | copertura completa dei writer |
| Stress (una tantum) | un run **512³** (o la taglia massima del nodo) confrontato np = 4 vs golden CPU np = 4 | percorsi di memoria grande, indici grandi, NumFiles > 1; intercetta overflow che i box piccoli non vedono |
| Calibrazione | banda np1-vs-np2 su 64³ e 128³ | le soglie devono valere a entrambe le risoluzioni; se la banda cambia molto con la griglia, le soglie in unità di cella vanno riviste |

Totale: ~3 × 2 × 2 = 12 run di validazione corrente (più stress e calibrazione una tantum) — sostenibile: a queste taglie PINOCCHIO gira in minuti.

### F.3 Manutenzione

- Il golden si rigenera **solo** quando cambia deliberatamente la base di riferimento (nuovo merge validato su master), mai per far passare un test. Ogni rigenerazione = nuovo manifest con nuovo commit, e il vecchio golden si archivia.
- `thresholds.json` è sotto versionamento; ogni modifica di soglia richiede una giustificazione nel commit message (quale misura l'ha motivata).
- Quando una metrica fallisce e l'indagine rivela un *falso allarme* strutturale (metrica mal posta, non soglia troppo stretta), si corregge la metrica e si annota qui il caso, come già fatto per "i conteggi non bastano" (FastFrag).
