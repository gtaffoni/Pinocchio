#!/usr/bin/env python3
"""
PINOCCHIO Multi-Agent Orchestrator
===================================
Coordina gli agenti specializzati per il redesign di PINOCCHIO.

Uso:
    python orchestrator.py --task merge
    python orchestrator.py --task propose-fragmentation-algorithms
    python orchestrator.py --task implement --branch feat/fragmentation-openmp
    python orchestrator.py --task review --branch feat/fragmentation-openmp
    python orchestrator.py --task document --file src/fragmentation.c
"""

import anthropic
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configurazione agenti
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "state"
AGENTS_DIR = BASE_DIR / "agents"

STATE_DIR.mkdir(exist_ok=True)

AGENTS = {
    "cosmologist": {
        "model": "claude-sonnet-4-20250514",
        "system_file": AGENTS_DIR / "cosmologist.md",
    },
    "hpc_expert": {
        "model": "claude-sonnet-4-20250514",
        "system_file": AGENTS_DIR / "hpc_expert.md",
    },
    "gpu_expert": {
        "model": "claude-sonnet-4-20250514",
        "system_file": AGENTS_DIR / "gpu_expert.md",
    },
    "reviewer": {
        "model": "claude-opus-4-20250514",      # Opus: review critico
        "system_file": AGENTS_DIR / "reviewer.md",
    },
    "documenter": {
        "model": "claude-haiku-4-5-20251001",   # Haiku: veloce per docs
        "system_file": AGENTS_DIR / "documenter.md",
    },
}

MAX_REVIEW_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()


def load_system_prompt(agent_name: str) -> str:
    path = AGENTS[agent_name]["system_file"]
    if not path.exists():
        raise FileNotFoundError(f"System prompt non trovato: {path}")
    return path.read_text()


def call_agent(agent_name: str, task: str, context: str = "",
               max_tokens: int = 8096) -> str:
    """Chiama un agente con un task e contesto opzionale."""
    cfg = AGENTS[agent_name]
    system = load_system_prompt(agent_name)
    
    content = task
    if context:
        content = f"## CONTESTO\n\n{context}\n\n## TASK\n\n{task}"
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
          f"Calling {agent_name} ({cfg['model']})...")
    
    response = client.messages.create(
        model=cfg["model"],
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    result = response.content[0].text
    
    # Log della chiamata
    log_agent_call(agent_name, task[:100], result[:200])
    return result


def log_agent_call(agent: str, task_summary: str, result_summary: str):
    log_file = STATE_DIR / "agent_calls.log"
    with open(log_file, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {agent}\n")
        f.write(f"  Task: {task_summary}...\n")
        f.write(f"  Result: {result_summary}...\n")


def read_source_files(src_dir: str, extensions: list = None) -> dict:
    """Legge i sorgenti da una directory."""
    if extensions is None:
        extensions = [".c", ".h", ".mk", "Makefile"]
    
    files = {}
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Warning: directory {src_dir} non trovata")
        return files
    
    for f in src_path.rglob("*"):
        if f.is_file() and (f.suffix in extensions or f.name in extensions):
            try:
                files[str(f.relative_to(src_path))] = f.read_text(errors='replace')
            except Exception as e:
                print(f"  Warning: non riesco a leggere {f}: {e}")
    return files


def git_create_branch(branch_name: str):
    """Crea un nuovo branch git da main."""
    try:
        subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-b", branch_name],
                      check=True, capture_output=True)
        print(f"  Branch creato: {branch_name}")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: git branch creation fallita: {e}")


def save_state(key: str, content: str):
    """Salva stato condiviso."""
    path = STATE_DIR / f"{key}.md"
    path.write_text(content)
    print(f"  Stato salvato: {path}")


def load_state(key: str) -> str:
    """Carica stato condiviso."""
    path = STATE_DIR / f"{key}.md"
    if path.exists():
        return path.read_text()
    return ""


# ---------------------------------------------------------------------------
# Task: Merge src/ + src_leonardo/
# ---------------------------------------------------------------------------

def task_merge():
    """Analizza e pianifica il merge dei due codebase."""
    print("\n=== TASK: MERGE src/ + src_leonardo/ ===")
    
    src_files = read_source_files("src")
    src_leo_files = read_source_files("src_leonardo")
    
    # Costruisci contesto per l'HPC Expert
    src_list = "\n".join(f"  - {f}" for f in sorted(src_files.keys()))
    leo_list = "\n".join(f"  - {f}" for f in sorted(src_leo_files.keys()))
    
    context = f"""
## File in src/
{src_list}

## File in src_leonardo/
{leo_list}

## Nota
src/ contiene la versione corrente con heFFTe + radix sort già integrati.
src_leonardo/ è la versione da mergere (potrebbe avere ottimizzazioni diverse).
Il Makefile deve essere preservato e funzionante dopo il merge.
"""
    
    # HPC Expert analizza le differenze
    analysis = call_agent(
        "hpc_expert",
        """Analizza le differenze tra src/ e src_leonardo/.
        
Per ogni file presente in entrambe le versioni, indica:
1. Se sono identici (o quasi)
2. Quali differenze significative ci sono
3. Quale versione preferire o come combinarle

Identifica anche:
- File presenti solo in una versione
- Dipendenze Makefile che cambiano
- Conflitti potenziali

Produci un piano di merge dettagliato e ordinato per priorità.""",
        context,
        max_tokens=8096
    )
    
    save_state("merge_analysis", analysis)
    print("\n--- ANALISI MERGE (salvata in state/merge_analysis.md) ---")
    print(analysis[:2000] + "..." if len(analysis) > 2000 else analysis)
    
    return analysis


# ---------------------------------------------------------------------------
# Task: Proposta algoritmica per frammentazione
# ---------------------------------------------------------------------------

def task_propose_fragmentation():
    """Il cosmologo propone algoritmi alternativi per la frammentazione."""
    print("\n=== TASK: PROPOSTA ALGORITMI FRAMMENTAZIONE ===")
    
    # Carica contesto esistente
    merge_analysis = load_state("merge_analysis")
    
    # Leggi il codice di frammentazione attuale
    frag_files = {}
    for src_dir in ["src", "src_leonardo"]:
        files = read_source_files(src_dir)
        for name, content in files.items():
            if "fragment" in name.lower() or "frag" in name.lower():
                frag_files[f"{src_dir}/{name}"] = content
    
    frag_context = "\n\n".join(
        f"### {name}\n```c\n{content[:3000]}\n```"
        for name, content in frag_files.items()
    )
    
    context = f"""
## Codice di frammentazione attuale
{frag_context if frag_context else "I file sorgente non sono ancora disponibili in questa directory. Ragiona sull'algoritmo dalla descrizione."}

## Stato del merge
{merge_analysis[:1000] if merge_analysis else "Merge non ancora analizzato."}
"""
    
    proposals = call_agent(
        "cosmologist",
        """Proponi metodi per migliorare e parallelizzare la frammentazione di PINOCCHIO.

Devi rispondere a due domande distinte:

**Domanda 1: Parallelizzazione OpenMP interna al rank**
Il loop seriale di frammentazione dentro ogni rank MPI può diventare OpenMP?
Analizza la struttura delle dipendenze dati e proponi se/come farlo.
Considera il wavefront parallelism basato su bin di z_c.

**Domanda 2: Metodi alternativi dalla letteratura**
Proponi 2-3 metodi alternativi di clustering/frammentazione che:
- Abbiano migliore scalabilità intrinseca
- Siano approssimati ma fisicamente motivati
- Siano implementabili in C con MPI+OpenMP

Per ogni metodo usa il formato standard del tuo system prompt.
Ogni proposta diventerà un branch git separato.""",
        context,
        max_tokens=8096
    )
    
    save_state("fragmentation_proposals", proposals)
    print("\n--- PROPOSTE (salvate in state/fragmentation_proposals.md) ---")
    print(proposals)
    
    return proposals


# ---------------------------------------------------------------------------
# Task: Implementazione di una proposta
# ---------------------------------------------------------------------------

def task_implement(branch_name: str, proposal_summary: str = None):
    """HPC Expert implementa una proposta specifica."""
    print(f"\n=== TASK: IMPLEMENTAZIONE su {branch_name} ===")
    
    if proposal_summary is None:
        proposals = load_state("fragmentation_proposals")
        proposal_summary = proposals[:3000] if proposals else "Nessuna proposta caricata"
    
    # Leggi codice corrente
    src_files = read_source_files("src")
    src_context = "\n\n".join(
        f"### src/{name}\n```c\n{content[:2000]}\n```"
        for name, content in list(src_files.items())[:5]  # limita a 5 file
    )
    
    context = f"""
## Proposta algoritmica
{proposal_summary}

## Codice corrente (estratto)
{src_context}
"""
    
    git_create_branch(branch_name)
    
    implementation = call_agent(
        "hpc_expert",
        f"""Implementa la proposta algoritmica per il branch {branch_name}.

L'implementazione deve:
1. Partire dal codice in src/ come base
2. Essere corretta prima di essere ottimizzata
3. Compilare con: clang -fopenmp -O2 (verificalo mentalmente)
4. Preservare il Makefile esistente
5. Includere commenti Doxygen base (il Documenter li raffinerà)

Fornisci il codice C completo per le funzioni modificate.""",
        context,
        max_tokens=8096
    )
    
    save_state(f"implementation_{branch_name.replace('/', '_')}", implementation)
    print("\n--- IMPLEMENTAZIONE ---")
    print(implementation[:3000] + "..." if len(implementation) > 3000 else implementation)
    
    return implementation


# ---------------------------------------------------------------------------
# Task: Review
# ---------------------------------------------------------------------------

def task_review(branch_name: str, implementation: str = None):
    """Reviewer (Opus) fa la review dell'implementazione."""
    print(f"\n=== TASK: REVIEW di {branch_name} ===")
    
    if implementation is None:
        key = f"implementation_{branch_name.replace('/', '_')}"
        implementation = load_state(key)
        if not implementation:
            print(f"Errore: nessuna implementazione trovata per {branch_name}")
            return None
    
    proposals = load_state("fragmentation_proposals")
    
    approved = False
    review_result = None
    
    for iteration in range(1, MAX_REVIEW_ITERATIONS + 1):
        print(f"\n  Iterazione review {iteration}/{MAX_REVIEW_ITERATIONS}")
        
        context = f"""
## Branch: {branch_name}
## Iterazione: {iteration}/{MAX_REVIEW_ITERATIONS}

## Proposta algoritmica originale (da verificare che sia implementata correttamente)
{proposals[:2000] if proposals else "Non disponibile"}

## Codice da revisionare
{implementation}
"""
        
        review_result = call_agent(
            "reviewer",
            f"Fai una review critica dell'implementazione nel branch {branch_name}. "
            f"Questa è l'iterazione {iteration} di {MAX_REVIEW_ITERATIONS}.",
            context,
            max_tokens=4096
        )
        
        save_state(f"review_{branch_name.replace('/', '_')}_iter{iteration}", review_result)
        print(review_result[:2000])
        
        if "APPROVED" in review_result and "CHANGES_REQUIRED" not in review_result:
            approved = True
            print(f"\n✅ APPROVED dopo {iteration} iterazione/i")
            break
        elif "ESCALATE" in review_result:
            print(f"\n⚠️  ESCALATE — problema strutturale, intervento umano necessario")
            break
        else:
            print(f"\n🔄 CHANGES_REQUIRED — torna all'HPC Expert")
            # In un sistema reale qui si richiamerebbe l'HPC Expert con le correzioni
            # Per ora logghiamo e usciamo
            break
    
    if not approved:
        save_state(f"review_log",
                  load_state("review_log") +
                  f"\n\n## {branch_name} — {datetime.now().isoformat()}\n"
                  f"Iterazioni: {iteration}\nEsito: {'APPROVED' if approved else 'PENDING'}\n")
    
    return review_result


# ---------------------------------------------------------------------------
# Task: Documentazione
# ---------------------------------------------------------------------------

def task_document(source_file: str):
    """Documenter (Haiku) genera/migliora la documentazione Doxygen."""
    print(f"\n=== TASK: DOCUMENTAZIONE di {source_file} ===")
    
    src_path = Path(source_file)
    if not src_path.exists():
        print(f"Errore: file non trovato: {source_file}")
        return None
    
    source_code = src_path.read_text(errors='replace')
    
    docs = call_agent(
        "documenter",
        f"""Genera documentazione Doxygen completa per il file {source_file}.

Per ogni funzione pubblica aggiungi o migliora la docstring.
Per ogni struct aggiungi documentazione dei campi.
Aggiungi l'intestazione del file se manca.

Restituisci il file completo con le docstring aggiunte/migliorate.""",
        f"```c\n{source_code}\n```",
        max_tokens=8096
    )
    
    # Salva versione documentata
    output_path = Path("docs") / src_path.name
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(docs)
    print(f"  Documentazione salvata in: {output_path}")
    
    return docs


# ---------------------------------------------------------------------------
# Workflow completo
# ---------------------------------------------------------------------------

def workflow_full_feature(feature_name: str, proposal_summary: str):
    """Workflow completo: proposta → implementazione → review → docs."""
    branch = f"feat/fragmentation-{feature_name}"
    
    print(f"\n{'='*60}")
    print(f"WORKFLOW COMPLETO: {feature_name}")
    print(f"Branch: {branch}")
    print(f"{'='*60}")
    
    # 1. Implementazione
    implementation = task_implement(branch, proposal_summary)
    
    # 2. GPU porting (opzionale, commentato per ora)
    # gpu_version = task_gpu_port(branch, implementation)
    
    # 3. Review
    review = task_review(branch, implementation)
    
    # 4. Documentazione (solo se approved)
    if review and "APPROVED" in review:
        # In un sistema reale scriveremmo prima il file, poi lo documentiamo
        print("\n  Documentazione: da eseguire sui file effettivi del branch")
    
    return {"branch": branch, "implementation": implementation, "review": review}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PINOCCHIO Multi-Agent Orchestrator")
    parser.add_argument("--task", required=True,
                       choices=["merge", "propose", "implement", "review",
                                "document", "full-workflow"],
                       help="Task da eseguire")
    parser.add_argument("--branch", default=None,
                       help="Nome del branch git (per implement/review)")
    parser.add_argument("--file", default=None,
                       help="File sorgente (per document)")
    parser.add_argument("--feature", default=None,
                       help="Nome della feature (per full-workflow)")
    
    args = parser.parse_args()
    
    if args.task == "merge":
        task_merge()
    elif args.task == "propose":
        task_propose_fragmentation()
    elif args.task == "implement":
        if not args.branch:
            print("Errore: --branch richiesto per implement")
            sys.exit(1)
        task_implement(args.branch)
    elif args.task == "review":
        if not args.branch:
            print("Errore: --branch richiesto per review")
            sys.exit(1)
        task_review(args.branch)
    elif args.task == "document":
        if not args.file:
            print("Errore: --file richiesto per document")
            sys.exit(1)
        task_document(args.file)
    elif args.task == "full-workflow":
        if not args.feature:
            print("Errore: --feature richiesto per full-workflow")
            sys.exit(1)
        proposals = load_state("fragmentation_proposals")
        workflow_full_feature(args.feature, proposals[:3000])


if __name__ == "__main__":
    main()
