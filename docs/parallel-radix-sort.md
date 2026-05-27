# Parallel Radix Sort e Direct Lookup Table per la Frammentazione di PINOCCHIO

> **Branch:** `feature/parallel-radix-sort`
> **Autori:** Giuliano Taffoni
> **Data:** Marzo 2026

---

## Indice

- [1. Motivazione](#1-motivazione)
- [2. Algoritmo: LSB Radix Sort](#2-algoritmo-lsb-radix-sort)
  - [2.1 Principio](#21-principio)
  - [2.2 Trasformazione delle chiavi](#22-trasformazione-delle-chiavi)
  - [2.3 Struttura di ogni passata](#23-struttura-di-ogni-passata)
  - [2.4 Double-Buffering](#24-double-buffering)
- [3. Direct Lookup Table](#3-direct-lookup-table)
  - [3.1 Problema originale](#31-problema-originale)
  - [3.2 Soluzione](#32-soluzione)
  - [3.3 Eliminazione dell'indirezione indices\[\]](#33-eliminazione-dellindirezione-indices)
- [4. Riuso degli Array Esistenti](#4-riuso-degli-array-esistenti-zero-allocazioni-aggiuntive)
- [5. File Modificati](#5-file-modificati)
- [6. Prestazioni Attese](#6-prestazioni-attese)
- [7. Correttezza e Compatibilita'](#7-correttezza-e-compatibilita)

---

## 1. Motivazione

Nella frammentazione di PINOCCHIO, le particelle devono essere ordinate per valore decrescente di $F_{\max}$ (tempo di collasso) prima di costruire i gruppi. Nel codice originale:

- **Ordinamento**: `qsort()` della libreria C standard — **O(N log N)**, seriale, con accesso indiretto alla memoria (double-dereference: `frag[indices[a]].Fmax`)
- **Ricerca particelle**: `bsearch()` in un array ordinato per posizione — **O(log N)** per ogni chiamata a `find_location()`

Per run a 100M+ particelle, l'ordinamento diventa un collo di bottiglia significativo: ~1.4 miliardi di confronti, ciascuno con pattern di accesso cache-unfriendly. Inoltre, `qsort` non e' parallelizzabile.

---

## 2. Algoritmo: LSB Radix Sort

### 2.1 Principio

Il radix sort e' un algoritmo di ordinamento **non comparativo** con complessita' **O(kN)**, dove $k$ e' il numero di cifre nella rappresentazione della chiave. Per chiavi a 32 bit con radice 8-bit (256 bucket), $k=4$ passate.

### 2.2 Trasformazione delle chiavi

Per ordinare valori floating-point in ordine decrescente, si sfrutta la rappresentazione IEEE 754:

```c
static inline unsigned int fmax_to_sortkey_desc(PRODFLOAT f)
{
    float ff = (float)f;           // troncamento a 32 bit (precisione sufficiente)
    unsigned int u;
    memcpy(&u, &ff, sizeof(u));    // reinterpretazione bit-a-bit
    unsigned int mask = -(u >> 31) | 0x80000000u;
    return ~(u ^ mask);            // NOT bit-a-bit -> ordine decrescente
}
```

I float IEEE 754 hanno la proprieta' che, dopo la trasformazione del bit di segno, il loro ordinamento come interi unsigned coincide con l'ordinamento numerico. Il `NOT` finale inverte l'ordine.

### 2.3 Struttura di ogni passata

Ogni passata (4 in totale) processa un byte della chiave a 32 bit ed e' composta da 3 fasi:

#### Fase 1 — Istogramma (parallela)

Ogni thread $T$ scansiona il proprio segmento contiguo di $N/P$ elementi e conta le occorrenze di ciascuno dei 256 valori di byte in un istogramma thread-locale:

```
hist[T][b] = numero di chiavi nel segmento di T con byte corrente = b
```

Nessuna sincronizzazione necessaria (buffer separati per thread).

#### Fase 2 — Prefix sum (seriale)

Calcolo degli offset globali iterando per bucket e poi per thread:

```c
for (b = 0; b < 256; b++)
    for (t = 0; t < P; t++) {
        offs[t][b] = running_sum;
        running_sum += hist[t][b];
    }
```

L'ordine `(bucket, thread)` garantisce la **stabilita'** dell'ordinamento: gli elementi del thread 0 precedono quelli del thread 1 all'interno di ogni bucket. Costo: $256 \times P$ operazioni (trascurabile).

#### Fase 3 — Scatter (parallela)

Ogni thread scrive i propri elementi nella posizione di destinazione usando gli offset precalcolati:

```c
dst[offs[tid][bucket]++] = src[i];
```

Nessun conflitto: ogni thread aggiorna solo i propri offset.

### 2.4 Double-Buffering

Si usano due coppie di array (chiavi + indici). Le passate pari leggono dall'array originale e scrivono nel buffer; le passate dispari fanno il contrario. Con 4 passate (numero pari), il risultato finale si trova nell'array originale — nessuna copia finale necessaria.

```
Passata 0: keys -> buf_keys,  indices -> buf_indices
Passata 1: buf_keys -> keys,  buf_indices -> indices
Passata 2: keys -> buf_keys,  indices -> buf_indices
Passata 3: buf_keys -> keys,  buf_indices -> indices
           ^^ risultato nell'array originale
```

---

## 3. Direct Lookup Table

### 3.1 Problema originale

Dopo l'ordinamento per $F_{\max}$, il codice deve localizzare le particelle vicine sulla griglia Lagrangiana. Nel codice originale:

1. Un **secondo ordinamento** (`qsort`) riordina per posizione sulla griglia
2. `find_location(i,j,k)` usa **binary search** (`bsearch`) nell'array `sorted_pos[]` — $O(\log N)$

### 3.2 Soluzione

Eliminare completamente il secondo ordinamento e trasformare `sorted_pos[]` in una **tabella di lookup diretto**:

```c
// Costruzione: O(N)
memset(sorted_pos, -1, Npart * sizeof(int));
for (i = 0; i < Nstored; i++)
    sorted_pos[frag_pos[i]] = i;    // posizione griglia -> indice nell'ordine Fmax
```

```c
// Lookup: O(1)
int find_location(int i, int j, int k)
{
    return sorted_pos[COORD_TO_INDEX(i, j, k, subbox.Lgwbl)];
}
```

Dove prima c'era:

```c
// Vecchia versione: O(log N)
int find_location(int i, int j, int k)
{
    int pos = COORD_TO_INDEX(i, j, k, subbox.Lgwbl);
    int *nn = bsearch(&pos, sorted_pos, Nstored, sizeof(int), compare_search);
    return (nn != NULL) ? nn - sorted_pos : -1;
}
```

**Vincolo**: `sorted_pos[]` deve avere dimensione `Npart` (non `Nstored`), quindi e' necessario che `Nalloc >= Npart`.

### 3.3 Eliminazione dell'indirezione `indices[]`

Con la tabella di lookup diretta, `find_location()` restituisce direttamente l'indice nell'ordine $F_{\max}$. Tutti i riferimenti successivi nel codice cambiano:

```c
// Prima:
neigh[nn] = group_ID[indices[pos]];
peak_cond &= (frag[iz].Fmax > frag[indices[pos]].Fmax);

// Dopo:
neigh[nn] = group_ID[pos];
peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);
```

Questo elimina un livello di indirezione in ogni accesso, migliorando il pattern di accesso alla cache.

---

## 4. Riuso degli Array Esistenti (Zero Allocazioni Aggiuntive)

L'implementazione riusa array gia' allocati come buffer temporanei per il radix sort:

| Array | Uso originale | Uso durante il sort | Ripristino |
|-------|--------------|-------------------|------------|
| `indices[Nalloc]` | Output: permutazione ordinata | Input/output del radix sort | Risultato del sort |
| `indicesY[Nalloc]` | Permutazione inversa temporanea | `buf_indices` (double-buffering) | Sovrascritto dopo per `reorder()` |
| `sorted_pos[Nalloc]` | Posizioni ordinate (vecchio) | `buf_keys` (double-buffering) | Sovrascritto con lookup table |
| `group_ID[Nalloc]` | Assegnamento gruppi (usato dopo) | `buf_idx` scratch | Azzerato con `memset` dopo il sort |

**Costo memoria aggiuntivo**: solo gli istogrammi temporanei per-thread sullo stack:

$$\text{nthreads} \times 256 \times \text{sizeof(int)} \times 2$$

Esempio: 16 thread = 32 KB — trascurabile.

---

## 5. File Modificati

### Nuovi file

| File | Righe | Contenuto |
|------|-------|-----------|
| `src/parallel_sort.h` | 68 | Header con prototipi e dichiarazioni inline per la trasformazione delle chiavi |
| `src/parallel_sort.c` | 296 | Implementazione del radix sort core e delle due funzioni pubbliche |

### File modificati

| File | Modifiche |
|------|-----------|
| `src/fragment.c` | Sostituzione di 2 chiamate `qsort` con `parallel_radix_sort_by_fmax_desc()`. Eliminazione del secondo sort per posizione. Costruzione della tabella di lookup diretto. Semplificazione di `find_location()`. Rimozione dell'indirezione `indices[]` nei confronti successivi. |
| `src/build_groups.c` | Rimozione dell'indirezione `indices[]` in 4 punti dove `find_location()` viene usato per accedere a `group_ID[]` e `frag[]`. |
| `src/allocations.c` | Rimozione del `#ifdef CLASSIC_FRAGMENTATION` attorno al controllo `Nalloc >= Npart`. Il vincolo e' ora richiesto in entrambe le modalita' per la lookup table. |
| `src/Makefile` | Aggiunta di `parallel_sort.o` alla lista `OBJECTS`. |

---

## 6. Prestazioni Attese

### Confronto della complessita'

| Operazione | Codice originale | Nuovo codice |
|------------|-----------------|--------------|
| Ordinamento per $F_{\max}$ | $O(N \log N)$, seriale | $O(4N)$, parallelo $P$ thread |
| Ordinamento per posizione | $O(N \log N)$, seriale | **Eliminato** |
| `find_location()` | $O(\log N)$ binary search | $O(1)$ lookup diretto |
| Accesso dati dopo find | `frag[indices[pos]]` (2 indirezioni) | `frag[pos]` (1 indirezione) |

### Speedup stimato

Per $N = 10^8$ particelle:

| Configurazione | Speedup ordinamento | Note |
|----------------|-------------------|------|
| 1 thread | 3-5x | Meno operazioni, migliore cache locality |
| 4 thread | 5-8x | Parallelismo + riduzione operazioni |
| 16 thread | 8-15x | Limitato dalla banda di memoria |
| 64+ thread | 10-20x | Saturazione bandwidth, prefix sum trascurabile |

Il lookup diretto $O(1)$ in `find_location()` produce un beneficio aggiuntivo proporzionale al numero di chiamate (una per ogni vicino di ogni particella processata), che scala linearmente con $N$.

### Fattori limitanti

1. **Prefix sum seriale**: $O(256 \times P)$ per passata — trascurabile fino a ~1000 thread
2. **Banda di memoria**: Con molti thread, la fase di scatter puo' saturare il bus memoria
3. **Overhead OpenMP**: Per $N < 10^6$, il costo di spawn/join dei thread domina

---

## 7. Correttezza e Compatibilita'

- L'algoritmo e' **stabile**: particelle con lo stesso $F_{\max}$ mantengono l'ordine relativo originale (grazie all'ordine `bucket -> thread` nel prefix sum)
- Compatibile con entrambe le modalita' `CLASSIC_FRAGMENTATION` e default
- Il risultato dell'ordinamento e' **identico** a quello di `qsort` (a meno di tie-breaking per particelle con $F_{\max}$ uguale, dove la stabilita' del radix sort puo' dare un ordine diverso ma ugualmente corretto)
- Il vincolo `Nalloc >= Npart` era gia' soddisfatto nella pratica per la maggior parte delle configurazioni; ora e' esplicito e verificato con un messaggio di errore chiaro
