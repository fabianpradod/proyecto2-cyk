# main.py
from __future__ import annotations
import argparse
from typing import Optional

from grammar_io import load_grammar, tokenize
from cnf import to_cnf
from cyk import cyk
from trees import build_trees, render_ascii

def main():
    parser = argparse.ArgumentParser(
        description="Proyecto 2"
    )
    parser.add_argument("--grammar", required=True, help="Ruta al archivo .txt de la gramática")
    parser.add_argument("--start", required=False, help="Símbolo inicial (opcional; por defecto el del archivo)")
    parser.add_argument("--sentence", required=False, help="Oración a evaluar, entre comillas")
    parser.add_argument("--print-tree", action="store_true", help="Imprimir árbol(es) de parseo en ASCII")
    parser.add_argument("--max-trees", type=int, default=1, help="Cuántos árboles imprimir (si hay ambigüedad)")
    args = parser.parse_args()

    G = load_grammar(args.grammar)  

    start = args.start or getattr(G, "start", None)
    if start is None:
        raise SystemExit("La gramática no tiene símbolo inicial definido")

    if args.sentence is None:
        raise SystemExit("Falta --sentence \"...\"")

    tokens = tokenize(args.sentence)

    cnf = to_cnf(G.productions, start)

    accepted, table, backpointers, elapsed_ms = cyk(tokens, cnf)

    print("SI" if accepted else "NO")
    print(f"tiempo_ms: {elapsed_ms:.2f}")

    if accepted and args.print_tree:
        trees = build_trees(backpointers, tokens, cnf.start)
        if not trees:
            print("(No se reconstruyeron árboles)")
            return
        for idx, t in enumerate(trees[: max(1, args.max_trees) ], start=1):
            print(f"\nÁrbol {idx}:")
            print(render_ascii(t))

if __name__ == "__main__":
    main()
