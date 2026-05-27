# Theoretical Background — Deep Reference

> Livello teorico completo. Leggi questo file SOLO quando il task richiede
> di ragionare sulla fisica del metodo o di modificare algoritmi fisici core.
> Per il lavoro ordinario usa `theoretical_background_quick.md`.

---

## 1. Fondamenti: Lagrangian Perturbation Theory (LPT)

### 1.1 Coordinate Lagrangiane vs Euleriane

In cosmologia si lavora con due sistemi di coordinate:
- **Euleriane** `x`: posizione corrente della particella
- **Lagrangiane** `q`: posizione iniziale (etichetta permanente)

La relazione tra i due è data dallo **spostamento** `Ψ(q,t)`:

```
x(q,t) = q + Ψ(q,t)
```

LPT espande `Ψ` in serie perturbativa nel parametro di crescita lineare `D(t)`:

```
Ψ = D·Ψ⁽¹⁾ + D²·Ψ⁽²⁾ + D³·Ψ⁽³⁾ + ...
```

- **1LPT** (Zel'dovich Approximation, ZA): `Ψ ≈ D·Ψ⁽¹⁾`
- **2LPT**: aggiunge termine quadratico, migliora descrizione collapsed regions
- **3LPT**: termine cubico, in produzione in Pinocchio

### 1.2 Zel'dovich Approximation

Al primo ordine, lo spostamento è proporzionale al gradiente del potenziale
peculiare `φ(q)`:

```
Ψ⁽¹⁾(q) = -∇_q φ(q)
```

La ZA descrive correttamente la formazione di strutture fino al primo
orbit crossing. Dopo, le traiettorie si intersecano (shell crossing) e
LPT diverge — questo è il punto di collasso.

### 1.3 Shear Tensor e Ellipsoidal Collapse

Il tensore di deformazione è la derivata seconda del potenziale:

```
ψ_ij(q) = ∂²φ/∂q_i ∂q_j
```

I suoi tre autovalori `λ₁ ≥ λ₂ ≥ λ₃` descrivono la deformazione
dell'ellissoide di massa lungo i tre assi principali.

Il **collasso ellissoidale** (Bond & Myers 1996, Monaco 1995, 1997)
usa questi autovalori per predire il tempo di collasso di ogni asse.
Un elemento di massa collassa (orbit crossing, OC) quando il primo
asse raggiunge dimensione zero, cioè quando il Jacobiano:

```
J = det|∂x/∂q| = 0
```

Questo definisce il **redshift di collasso** `z_c` per ogni particella.

---

## 2. Orbit Crossing e Collapse Times

### 2.1 Multi-scale smoothing

Il campo di densità `ρ(q)` viene convoluto con Gaussiane di ~20 raggi
di smoothing `R` logaritmicamente spaziati. Per ogni smoothing:

1. Si calcola il potenziale `φ(q,R)` via FFT
2. Si calcola il tensore di deformazione `ψ_ij(q,R)`
3. Si applica ellipsoidal collapse → si ottiene `z_c(q,R)`

Per ogni particella si registra:
- `z_c` = massimo su tutti gli smoothing (collasso più precoce)
- `R_c` = smoothing radius corrispondente
- `v_c` = velocità Zel'dovich a quel smoothing: `v_c ∝ ∇φ(q,R_c)`

### 2.2 Interpretazione fisica

Dopo `z_c`, processi non-lineari dominano e LPT non è più affidabile.
L'elemento di massa è un candidato per far parte di un alone collassato.
Può però finire in un filamento o sheet se le condizioni di accretion
non sono soddisfatte (vedi fragmentazione).

---

## 3. Fragmentazione

> Questo è il modulo oggetto del redesign. Comprendere l'algoritmo in
> dettaglio è essenziale per qualsiasi modifica a `fragment.c`.

### 3.1 Ordinamento

Le particelle vengono ordinate per `z_c` **decrescente** (collasso più
precoce prima). Il radix sort parallelo (`feature/parallel-radix-sort`)
ottimizza questo passo che era il principale collo di bottiglia prima
della fragmentazione stessa.

### 3.2 Algoritmo sequenziale (stato attuale)

Per ogni particella in ordine di `z_c` decrescente, si applica:

#### (1) Seed halo
Se la particella è un **massimo locale** di `z_c` tra i suoi 6 vicini
Lagrangiani (i 6 vertici adiacenti sulla griglia iniziale `q`), diventa
il seme di un nuovo alone.

#### (2) Accretion
Una particella accreta su un alone candidato (che contiene almeno un
suo vicino Lagrangiano) se la distanza `d` al tempo di collasso è:

```
d ≤ f_a · R_M + f_r
```

dove `R_M = M^(1/3)` è il "raggio" dell'alone in unità di grid spacing.
Se ci sono più aloni candidati, si sceglie quello con `d/R_M` minore.

#### (3) Merging
Se la particella ha più aloni candidati, questi si fondono se la loro
distanza reciproca soddisfa:

```
d ≤ f_m · R_M
```

dove `R_M` si riferisce all'alone più grande. Fino a 6 aloni possono
fondersi simultaneamente (raro; merger binari e ternari dominano).

#### (4) Filamenti
Particelle che non soddisfano le condizioni (2) e (3) vengono assegnate
al gruppo "filamenti". Per mimmare l'accrezione diretta dai filamenti
(osservata nelle simulazioni N-body), i vicini Lagrangiani di una
particella che accreta vengono anch'essi accreti se erano nel gruppo
filamenti.

#### (5) Parametri di calibrazione

| Parametro | Valore | Significato fisico |
|-----------|--------|--------------------|
| `f_a` | 0.18 | Linking length accretion (analogo a FOF) |
| `f_m` | 0.35 | Linking length merging |
| `f_r` | 0.7 | Correzione effetti di risoluzione griglia |

Calibrati su mass function FOF da simulazioni N-body (SCDM, ΛCDM 128³,
256³). Accordo entro ~1% sulla HMF. **Non modificare.**

### 3.3 Problema di parallelizzazione

L'algoritmo nella forma attuale è **intrinsecamente sequenziale**:
la decisione per la particella `i` dipende dallo stato degli aloni
costruiti dalle particelle `i-1, i-2, ...` (quelle con `z_c` maggiore).

Questa dipendenza sequenziale è il principale ostacolo al GPU offload.
Il redesign deve trovare una formulazione che:
1. Spezzi o riduca le dipendenze tra particelle vicine in `z_c`
2. Esponga parallelismo fine-grained sufficiente per GPU
3. Produca risultati numericamente equivalenti all'algoritmo seriale

Strategie da esplorare:
- **Wavefront parallelism**: particelle con `z_c` in un intervallo `[z, z+dz]`
  hanno dipendenze limitate — processarle in batch paralleli
- **Graph-based approach**: costruire il grafo delle dipendenze Lagrangiane
  e risolverlo con algoritmi paralleli (union-find, label propagation)
- **Approximate parallelism**: accettare piccole differenze statistiche
  (non numeriche) nella HMF se il guadagno in performance lo giustifica
  (da discutere esplicitamente con il team)

---

## 4. LPT in Pinocchio: dettagli implementativi

### 4.1 Calcolo via FFT

Lo spostamento LPT si calcola nello spazio di Fourier:

```
Ψ̃⁽¹⁾(k) = -ik/k² · δ̃(k)          (1LPT/ZA)
Ψ̃⁽²⁾(k) = termine quadratico in δ̃  (2LPT)
Ψ̃⁽³⁾(k) = termine cubico           (3LPT)
```

`δ̃(k)` è la trasformata di Fourier del campo di densità.
Le FFT sono il dominante computazionale in questa fase.

### 4.2 Backend FFT

| Backend | File | Stato | Note |
|---------|------|-------|------|
| FFTW3 | `fmax-fftw.c` | Legacy | CPU only |
| PFFT | `fmax-pfft.c` | Legacy | MPI decomposizione slab |
| heFFTe | `fmax-heffte.c` | ✅ Attivo (`refactoring_leonardo`) | GPU-portable, A100/H100 |

heFFTe (Highly Efficient FFT for Exascale) supporta:
- Backend: cuFFT (NVIDIA), rocFFT (AMD), oneMKL (Intel)
- Decomposizione pencil (più scalabile di slab per alto numero MPI ranks)
- Interfaccia C++ templata

### 4.3 Power Spectrum e condizioni iniziali

Il power spectrum `P(k)` viene letto da CAMB (`Pk_from_CAMB.c`) o
generato internamente. Il white noise può essere generato da `GenIC.c`
o letto da file esterno (`ReadWhiteNoise.c`).

---

## 5. Cosmologia implementata

### 5.1 Funzioni cosmologiche (cosmo.c)

- `H(z)`: parametro di Hubble
- `D(z)`: fattore di crescita lineare
- `f(z) = d ln D / d ln a`: tasso di crescita
- Spline precompilate in `def_splines.h` per performance

### 5.2 Modelli supportati (in produzione, fuori scope redesign)

- **ΛCDM standard**: baseline
- **Scale-dependent growth**: per neutrini massivi
- **Modified gravity**: parametrizzazione `f(R)` o simili
- **Read PK table**: power spectrum tabulato da CAMB

**Per il redesign corrente si assume ΛCDM standard.**

---

## 6. Validazione fisica

### 6.1 Halo Mass Function (HMF)

La mass function `n(M,z)` è il test primario di correttezza.
Confronto con fit analitici (Press-Schechter, Sheth-Tormen, Watson).
Script di validazione: `scripts/HMF_validation.py`.
Reference runs: `HMF_Validation/`, `tests/only_HMF_tests/`.

### 6.2 Power Spectrum

Confronto `P(k)` delle posizioni aloni con teoria lineare e N-body.
Reference: `tests/ICs_piti_vs_pinocchio/`, `tests/pk_and_HMF_tests/`.

### 6.3 Criterio di successo per il redesign

Qualsiasi implementazione GPU-parallel della fragmentazione deve
riprodurre la HMF dei run di riferimento entro la varianza statistica
del campione (non entro la precisione floating point, dato che
l'ordine di operazioni cambierà).
La tolleranza accettabile va concordata prima di iniziare i test.

---

## 7. Riferimenti bibliografici chiave

- Monaco (1995, 1997): ellipsoidal collapse come troncamento LPT
- Bond & Myers (1996): ellipsoidal collapse
- Zel'dovich (1970): 1LPT / Zel'dovich Approximation
- Monaco et al. (2002): paper originale Pinocchio
- Munari et al. (2017): versione con 3LPT
- Lippich et al. (2019): validazione e estensioni
