# PINOCCHIO Multi-Agent System — README

## Setup

```bash
# Installa dipendenze
pip install anthropic

# Metti i tuoi sorgenti nelle directory corrette
# (o crea symlink)
ln -s /path/to/pinocchio src
ln -s /path/to/pinocchio_leonardo src_leonardo

# Verifica che ANTHROPIC_API_KEY sia settata
export ANTHROPIC_API_KEY=sk-ant-...
```

## Workflow raccomandato

### Step 1: Merge dei due codebase
```bash
python orchestrator.py --task merge
# Leggi state/merge_analysis.md e implementa manualmente o lascia fare all'agente
```

### Step 2: Proposta algoritmica per la frammentazione
```bash
python orchestrator.py --task propose
# Leggi state/fragmentation_proposals.md
# Scegli quale proposta implementare per prima
```

### Step 3: Implementa una proposta specifica
```bash
python orchestrator.py --task implement --branch feat/fragmentation-openmp-wavefront
```

### Step 4: Review
```bash
python orchestrator.py --task review --branch feat/fragmentation-openmp-wavefront
```

### Step 5: Documentazione
```bash
python orchestrator.py --task document --file src/fragmentation.c
```

### Oppure tutto in uno (dopo la proposta)
```bash
python orchestrator.py --task full-workflow --feature openmp-wavefront
```

## Struttura dei branch

```
main                              ← stabile, non toccare
├── merge/unify-src               ← Step 1
├── feat/fragmentation-openmp     ← parallelizzazione OpenMP standard
├── feat/fragmentation-wavefront  ← wavefront parallelism su z_c bins
├── feat/fragmentation-unionfind  ← union-find parallelo
├── feat/fragmentation-approx-X   ← metodi approssimati dalla letteratura
└── feat/gpu-offload-frag         ← OpenMP target offload
```

## Modelli usati

| Agente | Modello | Perché |
|--------|---------|--------|
| cosmologist | claude-sonnet-4 | Ragionamento fisico profondo |
| hpc_expert | claude-sonnet-4 | Implementazione C complessa |
| gpu_expert | claude-sonnet-4 | Conoscenza GPU specializzata |
| reviewer | claude-opus-4 | Review critica, massima qualità |
| documenter | claude-haiku-4-5 | Veloce, economico, task strutturato |

## File di stato

Tutti in `state/`:
- `merge_analysis.md` — analisi differenze src/ vs src_leonardo/
- `fragmentation_proposals.md` — proposte algoritmiche del cosmologo
- `implementation_*.md` — implementazioni degli agenti HPC
- `review_*_iter*.md` — review iterative
- `review_log.md` — log globale delle review
- `benchmark_log.md` — risultati di performance (da aggiornare manualmente)
- `agent_calls.log` — log di tutte le chiamate agli agenti
