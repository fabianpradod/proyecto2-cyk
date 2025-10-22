"""
cnf.py
This module transforms CFG into CNF
Removing epsilon-productions, unit-productions, and useless symbols
ensuring remainings production are one of the two forms:

* A -> B C (binary non-terminal expansion)
* A -> a   (terminal production)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
import re
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

Symbol = str
ProductionRHS = Tuple[Symbol, ...]
Productions = Dict[Symbol, Iterable[ProductionRHS]]


@dataclass(frozen=True)
class CNFGrammar:
    """
    Container for the final CNF representation that the CYK algorithm expects

    Attributes
    ----------
    start:
        The start symbol (S0)
    unary:
        Mapping terminals (tokens) to the set of non-terminals that can
        produce that token directly
    binary:
        Mapping pairs of non-terminals to the set of parents that can
        produce that pair. Each key is a 2-tuple (B, C) representing the
        right-hand side of a CNF production A -> B C
    """

    start: Symbol
    unary: Dict[str, Set[Symbol]]
    binary: Dict[Tuple[Symbol, Symbol], Set[Symbol]]


def to_cnf(productions: Productions, start: Symbol) -> CNFGrammar:
    """
    Convert productions into CNF and return the dedicated CNFGrammar

    1. Augment the grammar with a start symbol
    2. Remove epsilon-productions
    3. Remove unit-productions
    4. Remove useless symbols (non-generating and unreachable)
    5. Replace terminals in long productions with dedicated pre-terminals
    6. Binarise any production whose right-hand side contains more than two symbols

    Parameters
    ----------
    productions:
        Mapping from non-terminals to the iterable of right-hand sides assigned
        to them
    start:
        Declared start symbol of the grammar before augmentation
    """

    working_prods: Dict[Symbol, Set[ProductionRHS]] = {
        lhs: set(map(tuple, rhs_list))
        for lhs, rhs_list in productions.items()
    }

    # If the provided start symbol has no productions the grammar is malformed
    if start not in working_prods:
        raise ValueError(
            f"The start symbol {start!r} does not appear on the left-hand side "
            "of any production."
        )

    # The algorithm starts by introducing a new start symbol that points
    # to the original one, ensuring that removing epsilon productions does
    # not accidentally delete entire language
    start_symbol = _fresh_symbol(base="S0", existing=set(working_prods))
    working_prods.setdefault(start_symbol, set()).add((start,))

    # eliminate epsilon productions
    nullable_nonterminals = _compute_nullable_nonterminals(working_prods)
    working_prods = _eliminate_epsilon_productions(
        working_prods, nullable_nonterminals
    )

    # eliminate unit productions (of the form A -> B)
    working_prods = _eliminate_unit_productions(working_prods)

    # eliminate useless symbols:
    # - remove non-generating symbols
    # - remove unreachable symbols
    working_prods = _eliminate_useless_symbols(working_prods, start_symbol)

    # convert the remaining productions to strict CNF shape
    cnf_prods = _binarise_and_isolate_terminals(working_prods)

    # materialise the CNFGrammar structure in the exact layout that the CYK algorithm uses
    unary_rules: DefaultDict[str, Set[Symbol]] = defaultdict(set)
    binary_rules: DefaultDict[Tuple[Symbol, Symbol], Set[Symbol]] = defaultdict(set)

    for lhs, rhs_set in cnf_prods.items():
        for rhs in rhs_set:
            if len(rhs) == 1:
                token = rhs[0]
                unary_rules[token].add(lhs)
            elif len(rhs) == 2:
                binary_rules[(rhs[0], rhs[1])].add(lhs)
            else:
                raise AssertionError(
                    f"Encountered non-CNF production {lhs} -> {rhs}; this "
                    "indicates a bug in the conversion pipeline."
                )

    return CNFGrammar(
        start=start_symbol,
        unary={token: set(parents) for token, parents in unary_rules.items()},
        binary={
            pair: set(parents) for pair, parents in binary_rules.items()
        },
    )


# Helper funs

def _compute_nullable_nonterminals(
    productions: Dict[Symbol, Set[ProductionRHS]]
) -> Set[Symbol]:
    """
    Gather the fixed-point set of nullable non-terminals

    A non-terminal is nullable if it can derive the empty string
    """

    nullable: Set[Symbol] = set()

    # add all non-terminals that explicitly contain an empty production.
    for lhs, rhs_set in productions.items():
        if () in rhs_set:
            nullable.add(lhs)

    # Keep expanding the set until no new nullable non-terminals are found
    changed = True
    while changed:
        changed = False
        for lhs, rhs_set in productions.items():
            if lhs in nullable:
                continue
            for rhs in rhs_set:
                if rhs and all(symbol in nullable for symbol in rhs):
                    nullable.add(lhs)
                    changed = True
                    break

    return nullable


def _eliminate_epsilon_productions(
    productions: Dict[Symbol, Set[ProductionRHS]],
    nullable: Set[Symbol],
) -> Dict[Symbol, Set[ProductionRHS]]:
    """
    Remove epsilon productions while keeping the language intact for non-empty strings
    """

    new_prods: Dict[Symbol, Set[ProductionRHS]] = {
        lhs: set() for lhs in productions
    }

    for lhs, rhs_set in productions.items():
        for rhs in rhs_set:
            if not rhs:
                continue

            # Generate all combinations by optionally removing nullable symbols.
            for expansion in _generate_nullable_expansions(rhs, nullable):
                if expansion:
                    new_prods[lhs].add(expansion)

    return new_prods


def _generate_nullable_expansions(
    rhs: ProductionRHS, nullable: Set[Symbol]
) -> Set[ProductionRHS]:
    """
    Generate every possible expansion of rhs by optionally dropping nullable non-terminals
    """

    expansions: Set[ProductionRHS] = set()
    total = len(rhs)

    def backtrack(index: int, buffer: List[Symbol]) -> None:
        if index == total:
            expansions.add(tuple(buffer))
            return

        symbol = rhs[index]

        # include the symbol as it is 
        buffer.append(symbol)
        backtrack(index + 1, buffer)
        buffer.pop()

        # drop the symbol if it is nullable
        if symbol in nullable:
            backtrack(index + 1, buffer)

    backtrack(0, [])
    return expansions


def _eliminate_unit_productions(
    productions: Dict[Symbol, Set[ProductionRHS]]
) -> Dict[Symbol, Set[ProductionRHS]]:
    """
    Replace unit productions A -> B with direct copies of B's non-unit productions
    """

    nonterminals = set(productions)

    # non_unit stores the productions that are already acceptable 
    # unit_graph records each A -> B unit edge so we can follow chains (A -> B -> C)
    non_unit: Dict[Symbol, Set[ProductionRHS]] = {
        lhs: set() for lhs in productions
    }
    unit_graph: Dict[Symbol, Set[Symbol]] = {lhs: set() for lhs in productions}

    for lhs, rhs_set in productions.items():
        for rhs in rhs_set:
            if len(rhs) == 1 and rhs[0] in nonterminals:
                unit_graph[lhs].add(rhs[0])
            else:
                non_unit[lhs].add(rhs)

    # Compute the closure of the unit graph using a BFS from each non-terminal
    result: Dict[Symbol, Set[ProductionRHS]] = {
        lhs: set(non_unit[lhs]) for lhs in productions
    }

    for origin in productions:
        queue: deque[Symbol] = deque(unit_graph[origin])
        visited: Set[Symbol] = set(unit_graph[origin])

        while queue:
            target = queue.popleft()
            # Copy all non-unit productions from the target into the origin
            result[origin].update(non_unit[target])

            for next_symbol in unit_graph[target]:
                if next_symbol not in visited:
                    visited.add(next_symbol)
                    queue.append(next_symbol)

    return result


def _eliminate_useless_symbols(
    productions: Dict[Symbol, Set[ProductionRHS]],
    start_symbol: Symbol,
) -> Dict[Symbol, Set[ProductionRHS]]:
    """
    Remove productions that involve non-generating or unreachable symbols.
    """

    # find generating symbols (those that can derive a string of terminals)
    nonterminals = set(productions)
    generating: Set[Symbol] = set()
    changed = True

    while changed:
        changed = False
        for lhs, rhs_set in productions.items():
            if lhs in generating:
                continue
            for rhs in rhs_set:
                if all(
                    symbol not in nonterminals or symbol in generating
                    for symbol in rhs
                ):
                    generating.add(lhs)
                    changed = True
                    break

    # Remove any production whose left-hand side is non-generating
    filtered: Dict[Symbol, Set[ProductionRHS]] = {}
    for lhs, rhs_set in productions.items():
        if lhs not in generating:
            continue
        valid_rhs: Set[ProductionRHS] = set()
        for rhs in rhs_set:
            if all(
                symbol not in nonterminals or symbol in generating
                for symbol in rhs
            ):
                valid_rhs.add(rhs)
        if valid_rhs:
            filtered[lhs] = valid_rhs

    # remove unreachable symbols using a BFS from the start symbol
    reachable: Set[Symbol] = set()
    queue: deque[Symbol] = deque([start_symbol])

    while queue:
        current = queue.popleft()
        if current not in filtered or current in reachable:
            continue
        reachable.add(current)

        for rhs in filtered[current]:
            for symbol in rhs:
                if symbol in filtered and symbol not in reachable:
                    queue.append(symbol)

    final_prods: Dict[Symbol, Set[ProductionRHS]] = {}
    for lhs, rhs_set in filtered.items():
        if lhs not in reachable:
            continue
        kept_rhs: Set[ProductionRHS] = set()
        for rhs in rhs_set:
            if all(
                symbol not in filtered or symbol in reachable
                for symbol in rhs
            ):
                kept_rhs.add(rhs)
        if kept_rhs:
            final_prods[lhs] = kept_rhs

    return final_prods


def _binarise_and_isolate_terminals(
    productions: Dict[Symbol, Set[ProductionRHS]]
) -> Dict[Symbol, Set[ProductionRHS]]:
    """
    Replace every production with an equivalent set that respects CNF shape
    """

    nonterminals: Set[Symbol] = set(productions)

    # The resulting productions are stored inside cnf_prods
    cnf_prods: Dict[Symbol, Set[ProductionRHS]] = {
        lhs: set() for lhs in productions
    }

    # terminal_cache ensures we reuse terminal symbols when the same terminal appear multiple times
    terminal_cache: Dict[str, Symbol] = {}

    # Counter used to generate unique fresh symbols
    counter = itertools.count(1)

    def ensure_preterminal(token: str) -> Symbol:
        """
        Return the non-terminal that produces token.  Create one if needed.
        """

        if token in terminal_cache:
            return terminal_cache[token]

        base_name = f"T_{_sanitize_token(token)}"
        candidate = _fresh_symbol(
            base=base_name,
            existing=nonterminals.union(terminal_cache.values()),
        )
        terminal_cache[token] = candidate
        nonterminals.add(candidate)
        cnf_prods.setdefault(candidate, set()).add((token,))
        return candidate

    def next_bin_symbol() -> Symbol:
        """
        Produce a unique symbol for binarisation steps.
        """

        base_name = f"BIN_{next(counter)}"
        candidate = _fresh_symbol(base=base_name, existing=nonterminals)
        nonterminals.add(candidate)
        return candidate

    for lhs, rhs_set in productions.items():
        for rhs in rhs_set:
            if len(rhs) == 1:
                symbol = rhs[0]
                if symbol in nonterminals:
                    #this case should have been removed by the unit-remover but we keep it incase
                    cnf_prods[lhs].add(rhs)
                else:
                    cnf_prods[lhs].add((symbol,))
            else:
                # Replace terminals occurring within longer productions with terminal non-terminals
                symbols: List[Symbol] = []
                for symbol in rhs:
                    if symbol in nonterminals:
                        symbols.append(symbol)
                    else:
                        symbols.append(ensure_preterminal(symbol))

                # Binarise the right-hand side
                current_lhs = lhs
                buffer = list(symbols)
                while len(buffer) > 2:
                    first, *rest = buffer
                    new_symbol = next_bin_symbol()
                    cnf_prods[current_lhs].add((first, new_symbol))
                    current_lhs = new_symbol
                    cnf_prods.setdefault(current_lhs, set())
                    buffer = rest
                cnf_prods.setdefault(current_lhs, set())
                cnf_prods[current_lhs].add(tuple(buffer))

    return cnf_prods


def _fresh_symbol(base: str, existing: Set[Symbol]) -> Symbol:
    """
    Generate a fresh symbol derived from base that does not collide with the ones listed
    """

    if base not in existing:
        return base

    index = 1
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _sanitize_token(token: str) -> str:
    """
    Turn token into an uppercase identifier-safe string.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").upper() or "TOKEN"
