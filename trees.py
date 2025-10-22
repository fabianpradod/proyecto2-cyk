"""
trees.py
This module reconstructs parse trees from the dynamic programming backpointer
table produced by the CYK algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Mapping, Sequence, Tuple, Union


Index = Tuple[int, int] 
Symbol = str
Token = str


@dataclass(frozen=True)
class ParseTree:
    """
    Immutable parse-tree node
    """

    label: Symbol
    children: Tuple[Union["ParseTree", Token], ...]

    def is_leaf(self) -> bool:
        return len(self.children) == 1 and isinstance(self.children[0], str)



def build_trees(
    backpointers: Sequence[Sequence[Mapping[Symbol, Iterable[object]]]],
    tokens: Sequence[Token],
    start_symbol: Symbol,
) -> List[ParseTree]:
    """
    Reconstruct all parse trees that derive "tokens" from "start_symbol"

    Parameters
    ----------
    backpointers:
        Upper-triangular CYK table stored as a sequence of sequences
    tokens:
        Tokenised input sentence
    start_symbol:
        Non-terminal that should span the entire sentence.

    Returns
    -------
    List[ParseTree]
        One parse tree per valid derivation
    """

    sentence_length = len(tokens)

    # Helper that looks up the mapping "non_terminal -> Iterable[descriptor]" 
    def get_cell(span_start: int, span_end: int):
        if span_end <= span_start:
            return {}
        try:
            row = backpointers[span_start]
            return row[span_end]
        except (IndexError, TypeError, KeyError):
            return {}

    @lru_cache(maxsize=None) # cool fun/decorator remembers cache of a past calls
    def expand(span_start: int, span_end: int, symbol: Symbol) -> Tuple[ParseTree, ...]:
        """
        Recursively expand "symbol" over "tokens[span_start:span_end]"
        """

        cell = get_cell(span_start, span_end)
        if not cell:
            return tuple()

        # The cell may be any mapping type 
        entries = cell[symbol] if symbol in cell else []

        found: List[ParseTree] = []
        for descriptor in entries:
            kind, data = _normalise_descriptor(descriptor)
            if kind == "terminal":
                token = data
                found.append(ParseTree(label=symbol, children=(token,)))
            elif kind == "binary":
                split, left_symbol, right_symbol = data
                left_trees = expand(span_start, split, left_symbol)
                right_trees = expand(split, span_end, right_symbol)
                for left_tree in left_trees:
                    for right_tree in right_trees:
                        found.append(
                            ParseTree(
                                label=symbol,
                                children=(left_tree, right_tree),
                            )
                        )

        # Deduplicate trees to avoid repeated derivations when multiple backpointers lead to identical structures
        unique_trees = tuple(dict.fromkeys(found))
        return unique_trees

    # Start the recursion from the root span.
    full_span_trees = expand(0, sentence_length, start_symbol)
    return list(full_span_trees)


def render_ascii(tree: ParseTree) -> str:
    """
    Render "tree" as a ASCII diagram
    """

    lines: List[str] = []

    def walk(node: Union[ParseTree, Token], prefix: str, is_last: bool) -> None:
        """
        Recursively append lines to "lines" in a depth-first traversal
        """

        connector = "`-- " if prefix else ""
        if isinstance(node, ParseTree):
            lines.append(f"{prefix}{connector}{node.label}")
            child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
            for index, child in enumerate(node.children):
                walk(child, child_prefix, index == len(node.children) - 1)
        else:
            lines.append(f"{prefix}{connector}{node}")

    walk(tree, prefix="", is_last=True)
    return "\n".join(lines)


# Normalisation helpers 

def _normalise_descriptor(descriptor: object) -> Tuple[str, Tuple]:
    """
    Convert different descriptor shapes into a canonical form
    """

    if isinstance(descriptor, dict):
        kind = descriptor.get("type") or descriptor.get("kind")
        if kind == "terminal":
            return "terminal", descriptor["token"]
        if kind == "binary":
            return "binary", (
                int(descriptor["split"]),
                descriptor["left"],
                descriptor["right"],
            )

    if isinstance(descriptor, tuple):
        if len(descriptor) == 2 and descriptor[0] == "terminal":
            return "terminal", descriptor[1]
        if len(descriptor) == 4 and descriptor[0] == "binary":
            return "binary", (int(descriptor[1]), descriptor[2], descriptor[3])

    kind_attr = getattr(descriptor, "type", None) or getattr(descriptor, "kind", None)
    if kind_attr == "terminal":
        token_value = getattr(descriptor, "token")
        return "terminal", token_value
    if kind_attr == "binary":
        return "binary", (
            int(getattr(descriptor, "split")),
            getattr(descriptor, "left"),
            getattr(descriptor, "right"),
        )

    raise ValueError(
        "Unsupported backpointer descriptor shape.  Expected either a mapping "
        "or a tuple describing a 'terminal' or 'binary' expansion."
    )
