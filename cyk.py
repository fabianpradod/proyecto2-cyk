# cyk.py
from __future__ import annotations
from typing import Dict, List, Mapping, Sequence, Set, Tuple
from time import perf_counter

from cnf import CNFGrammar  # ya existente

Symbol = str
Token = str

# Tipos de celdas: cada celda es un dict: no_terminal -> set() (solo presencia)
Cell = Dict[Symbol, Set[None]]
# Backpointers: mismo shape de tabla, pero mapean no_terminal -> List[descriptor]
BackptrCell = Dict[Symbol, List[object]]

def _empty_table(n: int) -> List[List[Cell]]:
    return [[{} for _ in range(n + 1)] for _ in range(n)]

def _empty_backptrs(n: int) -> List[List[BackptrCell]]:
    return [[{} for _ in range(n + 1)] for _ in range(n)]

def cyk(tokens: Sequence[Token], cnf: CNFGrammar
        ) -> Tuple[bool, List[List[Cell]], List[List[BackptrCell]], float]:
    """
    Ejecuta CYK sobre tokens usando una gramática en CNF.

    Devuelve:
        accepted: bool
        table: tabla CYK (presencias)
        backpointers: para reconstrucción de árboles
        elapsed_ms: float (ms)
    """
    n = len(tokens)
    table: List[List[Cell]] = _empty_table(n)
    backpointers: List[List[BackptrCell]] = _empty_backptrs(n)

    t0 = perf_counter()

    # Relleno de long = 1 (hojas)
    for i, tok in enumerate(tokens):
        parents = cnf.unary.get(tok, set())
        if not parents:
            # si el token no existe en la CNF, la celda queda vacía
            continue
        cell = table[i][i + 1] = {}
        bp_cell = backpointers[i][i + 1] = {}
        for A in parents:
            cell.setdefault(A, set()).add(None)
            bp_cell.setdefault(A, []).append({"type": "terminal", "token": tok})

    # Spans más largos
    for span_len in range(2, n + 1):           # longitud del span
        for i in range(0, n - span_len + 1):   # inicio
            j = i + span_len                   # fin (exclusivo)
            cell = table[i][j] = {}
            bp_cell = backpointers[i][j] = {}
            # Todos los splits posibles
            for k in range(i + 1, j):
                left_cell = table[i][k]
                right_cell = table[k][j]
                if not left_cell or not right_cell:
                    continue
                # Emparejar B en izquierda con C en derecha
                for B in left_cell.keys():
                    for C in right_cell.keys():
                        parents = cnf.binary.get((B, C), set())
                        if not parents:
                            continue
                        for A in parents:
                            cell.setdefault(A, set()).add(None)
                            bp_cell.setdefault(A, []).append({
                                "type": "binary",
                                "split": k,
                                "left": B,
                                "right": C,
                            })

    elapsed_ms = (perf_counter() - t0) * 1000.0
    accepted = cnf.start in table[0][n] if n > 0 else False
    return accepted, table, backpointers, elapsed_ms
