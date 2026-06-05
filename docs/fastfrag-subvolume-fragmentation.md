# FastFrag: Frammentazione per Sotto-Volumi

## Cos'è FastFrag?

FastFrag è una reimplementazione dell'algoritmo di frammentazione di PINOCCHIO sviluppata da un
collaboratore (directory `src_fastfrag/`). L'idea centrale è:

> **Dividere il dominio spaziale in piccoli sotto-volumi indipendenti, frammentare ciascuno
> separatamente, poi riconciliare i cataloghi.**

Questo approccio mira a spezzare la dipendenza globale che rende il loop standard intrinsecamente
seriale — e, in prospettiva, a permettere l'offloading su GPU di sotto-volumi interi.

---

## Perché il loop standard è seriale

Prima di capire FastFrag, è utile capire perché il loop di frammentazione standard non si
parallelizza facilmente.

Le particelle vengono processate in ordine **decrescente di Fmax = 1 + z_c** (da quelle collassate
prima a quelle collassate dopo). Quando si processa la particella P:

- I suoi 6 vicini lagrangiani con `Fmax > F` sono già stati processati → il loro `group_ID` è noto
- I vicini con `Fmax < F` non ancora processati → `group_ID = 0`

La decisione di P (è un peak? a quale gruppo si accreta?) dipende dallo stato corrente dei vicini.
L'ordine di processing non è arbitrario: è dettato dalla fisica (le strutture più dense collassano
prima). Cambiare l'ordine cambia il risultato.

Questo crea una **dipendenza seriale globale**: ogni particella dipende da tutte quelle con Fmax
maggiore, le quali dipendono a loro volta dalle particelle precedenti, etc.

---

## L'idea di FastFrag: sotto-volumi indipendenti

### Intuizione fisica

Immaginiamo di suddividere il dominio in piccoli cubi. Un alone che si forma "nel mezzo" di un cubo
non interagisce con aloni che si formano in cubi lontani (i vicini lagrangiani distano al massimo
1 grid spacing). I cubi sufficientemente piccoli possono essere frammentati **indipendentemente**.

La dipendenza esiste solo ai **bordi** tra sotto-volumi adiacenti. FastFrag gestisce questa
complicazione con un approccio a più passate.

### Decomposizione in Nsub³ sotto-volumi

Il dominio locale al task MPI viene suddiviso in `Nsub³` sotto-volumi cubici.
Con `Nsub = 4` si ottengono `4³ = 64` sotto-volumi.

**Esempio 2D con Nsub=4 su un dominio 16×16:**

```
 +----+----+----+----+
 | S0 | S1 | S2 | S3 |   ← Nsub=4 colonne
 +----+----+----+----+
 | S4 | S5 | S6 | S7 |   Ogni blocco è un sotto-volume 4×4
 +----+----+----+----+
 | S8 | S9 |S10 |S11 |
 +----+----+----+----+
 |S12 |S13 |S14 |S15 |
 +----+----+----+----+
```

Ogni sotto-volume contiene `(dominio/Nsub)³` particelle. Per un dominio 128³ con Nsub=4:
ogni sotto-volume è 32³ = 32.768 particelle.

### Frammentazione indipendente per sotto-volume

Ogni sotto-volume viene frammentato con il **medesimo algoritmo standard** (ordine Fmax decrescente,
peak/accretion/merge), ma operando su array **locali al sotto-volume**:

```c
typedef struct {
    int Npeaks, Ngroups, Npart;
    int GridSize[3];   // dimensioni del sotto-volume
    int Start[3];      // coordinate lagrangiane di inizio nel dominio globale
    product_data *Frag;         // dati di collapse (come frag[] globale)
    group_data   *Groups;       // catalogo aloni locali
    int          *Group_ID;     // assegnazione particella→alone locale
    int          *Linking_list; // lista concatenata
} volume_data;
```

Il vantaggio: ogni sotto-volume è **autocontenuto**. Non legge né scrive array globali.
Due sotto-volumi non adiacenti possono essere frammentati **in parallelo**.

### Il problema dei bordi

Un alone che si trova vicino al bordo di un sotto-volume può avere vicini lagrangiani
nell'altro sotto-volume. Se i due sotto-volumi sono frammentati separatamente, quell'alone
viene visto come "troncato" — manca informazione sulla metà dei vicini.

**Esempio 2D — alone a cavallo del bordo:**

```
Sotto-volume A    |    Sotto-volume B
  . . . . X X X  |  X X X . . . .
  . . . X X X .  |  . X X . . . .
  . . . . X X .  |  . X . . . . .
  . . . . . . .  |  . . . . . . .
                  ↑
               Bordo
```

Le particelle marcate con X appartengono a un unico alone fisico. Se A e B vengono frammentati
separatamente, A trova un alone piccolo e B trova un altro alone piccolo. La riconciliazione
deve capire che sono lo stesso alone.

---

## Le 8 passate con sfasamento progressivo

Questo è il cuore dell'algoritmo FastFrag. Il problema dei bordi viene risolto **ripetendo la
frammentazione 8 volte con griglie di sotto-volumi sfasate**.

### Logica dello sfasamento

Con `size = dominio/Nsub` (lato di ogni sotto-volume), le 8 passate usano gli offset:

| Passata | Offset (x, y, z) | Descrizione |
|---------|-----------------|-------------|
| 1 | (0, 0, 0) | Griglia originale |
| 2 | (size/2, 0, 0) | Sfasata di metà lungo x |
| 3 | (0, size/2, 0) | Sfasata di metà lungo y |
| 4 | (0, 0, size/2) | Sfasata di metà lungo z |
| 5 | (size/2, size/2, 0) | Sfasata lungo x e y |
| 6 | (size/2, 0, size/2) | Sfasata lungo x e z |
| 7 | (0, size/2, size/2) | Sfasata lungo y e z |
| 8 | (size/2, size/2, size/2) | Sfasata su tutti e tre gli assi |

### Visualizzazione in 2D

Consideriamo un dominio 1D semplificato di lunghezza 16, Nsub=4, size=4.

**Passata 1 (offset=0):** bordi a posizioni 0, 4, 8, 12
```
|----A----|----B----|----C----|----D----|
0         4         8        12        16
```

**Passata 2 (offset=2=size/2):** bordi a posizioni 2, 6, 10, 14
```
  |----E----|----F----|----G----|----H...|
  2         6        10        14       16
```

Un punto in posizione 3 (vicino al bordo di A nella Passata 1) si trova al **centro** del
sotto-volume E nella Passata 2. Il bordo di E è a distanza 1 (posizione 2), quello di F
è a distanza 3 (posizione 6). Con `size/2 = 2`, il punto è **lontano dai bordi**.

### La proprietà chiave: ogni punto è "al centro" in almeno una passata

Con le 8 passate, ogni punto del dominio si trova a **distanza ≥ size/4 dal bordo** di
almeno uno dei sotto-volumi nelle 8 configurazioni.

In 3D, ogni punto si trova vicino a un bordo in al massimo 3 direzioni simultaneamente.
Le 2³=8 combinazioni di sfasamento coprono tutte le situazioni.

**Esempio 2D — punto P a posizioni (3, 3) con size=4:**

```
Pass 1 (0,0):    bordi a x=0,4,8,...  e y=0,4,8,...
                 P a distanza 1 dal bordo x (a 4), distanza 3 dal bordo y (a 0)
                 → P è VICINO al bordo x

Pass 2 (2,0):    bordi a x=2,6,...  e y=0,4,8,...
                 P a distanza 1 dal bordo x (a 2), distanza 3 dal bordo y (a 0)
                 → P è ancora VICINO al bordo x

Pass 3 (0,2):    bordi a x=0,4,8,...  e y=2,6,...
                 P a distanza 1 dal bordo x (a 4), distanza 1 dal bordo y (a 2)
                 → P è VICINO a entrambi i bordi

Pass 4 (2,2):    bordi a x=2,6,...  e y=2,6,...
                 P a distanza 1 dal bordo x (a 2), distanza 1 dal bordo y (a 2)
                 → P è VICINO a entrambi i bordi... MA il sotto-volume va da (2,2) a (6,6)!
                 → Rispetto al centro del sotto-volume (4,4), P è a distanza (1,1)
                 → P è nel QUADRANTE INTERNO del sotto-volume
```

In questo ultimo caso, P ha tutti i vicini nello stesso sotto-volume → frammentazione corretta.

---

## Criteri di affidabilità: R_THR e A_THR

Non ogni frammentazione di sotto-volume è affidabile. Un alone vicino al bordo del suo
sotto-volume potrebbe avere vicini importanti "tagliati fuori".

FastFrag usa due criteri per determinare se un alone è stato risolto **correttamente**:

### Criterio A_THR (distanza assoluta)

```c
#define A_THR 9  // celle di griglia
```

Se l'alone è a distanza ≥ A_THR celle dal bordo del sotto-volume in **tutte e 3 le direzioni**,
si considera completamente risolto — a prescindere dalla sua massa.

**Motivazione:** i vicini lagrangiani distano 1 cella. Un alone a 9 celle dal bordo ha tutti i
vicini entro il sotto-volume con ampio margine.

### Criterio R_THR (distanza in unità di raggio lagrangiano)

```c
#define R_THR 2  // multipli del raggio lagrangiano
```

Il raggio lagrangiano di un alone di `M` particelle è `R_L = M^(1/3)` (in unità di grid spacing).
Un alone è "sicuro" se:

```
distanza_min_dal_bordo > R_THR × M^(1/3)
```

Con R_THR=2, un alone da 1000 particelle (R_L≈10) richiede distanza > 20 celle dal bordo.
Un alone da 8 particelle (R_L=2) richiede distanza > 4 celle.

**Motivazione:** un alone massivo ha una sfera di influenza (accretion radius) proporzionale a
`f_a × M^(1/3) ≈ 0.18 × M^(1/3)`. Per essere sicuri che nessuna particella che dovrebbe
accretarsi stia fuori dal sotto-volume, serve margine proporzionale a R_L.

### Un alone è "risolto" se:

```
distanza_min_dal_bordo > A_THR
    OPPURE
distanza_min_dal_bordo > R_THR × M^(1/3)
```

---

## Struttura di una passata

Ogni passata dell'algoritmo FastFrag procede come segue:

```
1. Costruisci la griglia di sotto-volumi con l'offset della passata corrente
2. Per ogni sotto-volume V (in parallelo, nella versione finale):
   a. initialize_volume(V): copia le particelle di V nel buffer locale v->Frag[]
   b. build_groups_in_volume(V): frammenta V con l'algoritmo standard
   c. merge_catalogs(V): riconcilia il catalogo locale con groups[] globale
3. Marca come "risolti" gli aloni che soddisfano R_THR o A_THR
```

### merge_catalogs() in dettaglio

Questa è la parte più delicata. Quando un sotto-volume finisce la sua frammentazione,
ha trovato aloni locali con IDs `1, 2, 3, ...`. Questi devono essere mappati agli aloni
globali in `groups[]`.

La logica è:
- Per ogni alone locale, trova il corrispondente alone globale (se esiste) confrontando le
  posizioni dei picchi (massimi locali di Fmax)
- Se l'alone locale corrisponde a un alone globale già noto → unisci i cataloghi
- Se l'alone locale è nuovo (picco non visto prima) → alloca un nuovo entry in groups[]
- Le particelle che nei run precedenti erano in un alone diverso vengono riassegnate

Questa riconciliazione introduce un certo overhead e una fonte di approssimazione: se lo stesso
picco fisico appare come due picchi distinti in due passate diverse (perché visto in contesti di
vicini diversi), la riconciliazione può sbagliare.

---

## La passata finale per aloni "irrisolti"

Dopo le 8 passate standard, alcuni aloni rimangono marcati come "irrisolti": sono troppo vicini
ai bordi in tutte le 8 configurazioni di griglia. Questo accade tipicamente per aloni molto
massivi (R_L grande) o in aree particolarmente dense.

Per questi aloni, FastFrag fa una **passata finale adattiva**:

```
Per ogni alone non ancora risolto:
  1. Costruisci un sotto-volume centrato sul picco dell'alone
  2. Dimensione iniziale: size × 1.5 (50% più grande del normale)
  3. Frammenta questo sotto-volume
  4. Controlla se l'alone ora soddisfa R_THR o A_THR
  5. Se no: espandi il sotto-volume e riprova (fino a coprire tutto il dominio)
```

Questo garantisce che **ogni alone venga risolto correttamente** alla fine, anche se richiede
frammentare sotto-volumi più grandi o addirittura l'intero dominio per gli aloni più massicci.

---

## Relazione con la decomposizione MPI

FastFrag opera **all'interno di ogni task MPI**, esattamente come il loop standard.
Ogni rank MPI possiede una "subbox" del dominio globale. La decomposizione in sotto-volumi
avviene all'interno della subbox locale.

```
Dominio globale (es. 128³)
├── Rank 0: subbox 64×64×64 (x: 0-63)
│   ├── Sotto-volume [0,0,0] → [31,31,31]
│   ├── Sotto-volume [32,0,0] → [63,31,31]
│   ├── ... (64 sotto-volumi in totale con Nsub=4 su 64³)
├── Rank 1: subbox 64×64×64 (x: 64-127)
│   ├── ... stessa struttura
```

Le particelle ai bordi tra rank MPI vengono scambiate **prima** della frammentazione (fase di
redistribuzione, già esistente in PINOCCHIO). Questo garantisce che ogni rank abbia una visione
completa dei vicini lagrangiani ai suoi bordi — esattamente come nel codice originale.

> **Nota bug nell'implementazione corrente:** in `src_fastfrag/fragment.c` riga 208,
> `size = MyGrids[0].GSglobal[0]/Nsub` usa la dimensione della griglia **globale** invece
> della subbox locale. Per un run con 4 MPI task su 128³, il rank vede una subbox 64³ ma
> calcola `size = 128/4 = 32`, il che è corretto per caso (32 = 64/2 ≈ subbox/Nsub con Nsub
> effettivo 2). Con layout MPI diversi questo darebbe risultati sbagliati.

---

## Potenziale per il parallelismo

### Parallelismo OpenMP (CPU)

I sotto-volumi che non condividono bordi possono essere frammentati **in parallelo**:

```c
#pragma omp parallel for
for (int v = 0; v < Nsub*Nsub*Nsub; v++) {
    initialize_volume(&volumes[v]);
    build_groups_in_volume(&volumes[v]);
    // merge_catalogs NON è parallelizzabile direttamente (scrive in groups[] globale)
}
// Fase di merge seriale
for (int v = 0; v < Nsub*Nsub*Nsub; v++) {
    merge_catalogs(&volumes[v]);
}
```

Il `merge_catalogs()` rimane seriale (accede a `groups[]` globale), ma la frammentazione
di ciascun sotto-volume è completamente indipendente.

### Parallelismo GPU (OpenMP target offload)

La struttura `volume_data` è ideale per il GPU offloading. Ogni sotto-volume è una struttura
autocontenuta che può essere mappata su device con:

```c
#pragma omp target map(to: v->Frag[0:v->Npart]) \
                   map(tofrom: v->Groups[0:v->Ngroups]) \
                   map(tofrom: v->Group_ID[0:v->Npart]) \
                   map(tofrom: v->Linking_list[0:v->Npart])
{
    build_groups_in_volume(v);
}
```

Il vantaggio rispetto al tiling: il tiling opera su array globali condivisi (`frag[]`, `groups[]`,
`group_ID[]`) — difficili da mappare su GPU in modo efficiente. FastFrag isola ogni
sotto-volume in memoria separata → trasferimento dati pulito, nessun accesso a memoria host
durante l'esecuzione.

### Limite fondamentale rimasto

Anche con FastFrag, il **loop interno** di `build_groups_in_volume()` rimane seriale nell'ordine
Fmax. Un singolo sotto-volume da 32³ ≈ 32.768 particelle viene processato sequenzialmente da
un thread/warp. L'unico parallelismo sfruttabile è tra sotto-volumi distinti.

Per una GPU moderna con migliaia di core, sarebbe necessario parallelizzare anche il loop interno —
il che richiederebbe un cambio algoritmico più radicale (es. algoritmi di tipo union-find parallelo).

---

## Confronto: Tiling 8-colori vs FastFrag

### Schema architetturale a confronto

**Tiling 8-colori (implementato in `src/build_groups.c`):**

```
Loop su 8 colori (sequenziale)
  └─ Loop parallelo (OMP) su tiles dello stesso colore
       └─ Loop su particelle nel tile (sequenziale, ordine Fmax)
```

**FastFrag:**

```
Loop su 8 passate con offset (sequenziale)
  └─ Loop parallelo (OMP) su sotto-volumi (in sviluppo)
       └─ build_groups_in_volume(): loop Fmax completo (sequenziale)
  └─ merge_catalogs() (seriale)
```

### Tabella comparativa

| Aspetto | Tiling 8-colori | FastFrag |
|---------|----------------|----------|
| **Stato** | Implementato, funzionante | Prototipo incompleto |
| **Correttezza** | Garantita per TILE_SIZE > 2×f_a×M_max^(1/3) | Approssimata (dipende da R_THR, A_THR) |
| **Differenza vs seriale** | ~1.8% good halos in più (128³) | Da verificare |
| **Granularità parallela** | Tile = TILE_SIZE³ = 512 particelle | Volume = (dom/Nsub)³ ≈ 32K particelle |
| **Overhead sinc.** | 8 barrier OMP (una per colore) | 8 loop su Nsub³ merge_catalogs |
| **Array globali** | Sì — tutti i thread accedono a frag[], groups[] | No — ogni sotto-volume ha propri array |
| **GPU friendliness** | Bassa (array globali condivisi) | Alta (volume_data autocontenuto) |
| **Aloni massivi** | Gestiti se TILE_SIZE adeguato | Richiedono passata finale adattiva |
| **Complessità codice** | Moderata | Alta (merge_catalogs è complesso) |
| **Bug noti** | Nessuno critico | size usa GSglobal invece di subbox |

### Quale scegliere per quale scopo?

**Per parallelismo CPU OMP a breve termine → Tiling 8-colori**
- Già implementato e funzionante
- Correttezza garantita per i parametri standard di PINOCCHIO
- Overhead basso (solo 8 barrier)
- Non richiede step di merge/riconciliazione

**Per GPU offloading a lungo termine → FastFrag (con revisione)**
- La struttura `volume_data` è il design pattern corretto per GPU
- Ogni sotto-volume mappa 1:1 su una GPU (o su un blocco di thread GPU)
- Richiede correzione del bug size, completamento di merge_catalogs, validazione fisica
- Il loop interno rimane seriale anche su GPU → ulteriore ricerca necessaria

**Posizione raccomandata:**
Il tiling 8-colori è la via pragmatica per guadagni immediati su CPU. FastFrag è il prototipo
per il redesign GPU-oriented futuro. I due approcci non si escludono: il tiling può essere
visto come un FastFrag con sotto-volumi = tile, ma senza step di merge (grazie alla garanzia
di non-interferenza dei colori). Una possibile convergenza è usare la struttura `volume_data`
di FastFrag con la garanzia di correttezza del tiling (volume grande abbastanza da evitare
interferenze).

---

## Stato dell'implementazione in `src_fastfrag/`

Il codice in `src_fastfrag/` è un **prototipo di ricerca** — funzionante ma incompleto:

- `fragment.c`: framework principale con le 8 passate (funzionante)
- `build_groups.c`: `build_groups_in_volume()` adattato per `volume_data` (funzionante)
- `build_groups_from_peaks.c`: tentativo alternativo di merger cinematico (ABORTITO, non usare)
- `fragment.h`: definizioni struct `volume_data`, `pos_data`

**TODO dal file `src_fastfrag/TODO`:**
```
V trovare bug
V confrontarsi con la frammentazione standard e classic su 8 task
-> A_THR meglio metterlo a 5 o 6, ideale 10
-> R_THR inutile metterlo oltre 2
CON R_THR>8 CRASHA, TROVARE IL BUG (crasha anche con Nsub=2)
```

Il crash con R_THR>8 suggerisce un problema di memoria nei buffer dei sotto-volumi.
Prima di integrare FastFrag in `src/`, questi bug devono essere risolti.
