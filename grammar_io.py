"""
grammar_io.py
This module is in charge of reading the context-free grammar description that is
stored in a .txt file and normalising it into the in-memory structure that
the rest of the project expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple


Symbol = str
# ProductionRHS represents the right-hand side of a production 
ProductionRHS = Tuple[Symbol, ...]
# Productions maps each non-terminal (left-hand side) to the list of expansions 
Productions = Dict[Symbol, List[ProductionRHS]]


@dataclass(frozen=True)
class Grammar:
    """
    Simple container that carries both the start symbol and the productions
    """

    start: Symbol
    productions: Productions

# _TOKEN_RE captures alphabetic words that can include an apostrophe in the middle (don't)
# re.findall with this pattern returns the tokens without punctuation
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# _COMMENT_RE matches anything after a hash symbol, allowing inline comments inside the grammar file
_COMMENT_RE = re.compile(r"#.*$")


def load_grammar(path: str) -> Grammar:
    """
    Read a grammar definition from path and normalise it.

    Parameters
    ----------
    path:
        Path to the plain-text grammar description

        * Empty lines are ignored.
        * Everything that follows a # is considered a comment.
        * A single optional line can define the start symbol explicitly using
          start: <symbol> (case sensitive).
        * Productions use the ASCII arrow -> (the loader also replaces the
          Unicode right-arrow → automatically)
          Example:

              S -> NP VP
              NP -> Det N | he | she
              Det -> a | the

        * Tokens can optionally be wrapped in single or double quotes, quotes are stripped during normalisation
    Returns
    -------
    Grammar
        Dataclass that contains both the detected start symbol and the
        production mapping.
    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the file does not declare any production or contains malformed
        lines.
    """
    file_path = Path(path)

    raw_text = file_path.read_text(encoding="utf-8")

    # productions collects the parsed rules.  We store them as lists because
    # the conversion to CNF may reorder or expand them.  Later stages will
    # deduplicate if needed.
    productions: Productions = {}

    # start_symbol will either be explicitly set by a "start: line" or default to the left-hand side of the first production
    start_symbol: Symbol | None = None

    for line_number, original_line in enumerate(raw_text.splitlines(), start=1):
        cleaned = _COMMENT_RE.sub("", original_line).strip()
        if not cleaned:
            continue

        cleaned = cleaned.replace("→", "->")

        if cleaned.lower().startswith("start:"):
            candidate = cleaned.split(":", 1)[1].strip()
            if not candidate:
                raise ValueError(
                    f"Line {line_number}: 'start:' must be followed by a non-terminal symbol."
                )
            start_symbol = candidate
            continue

        # All remaining lines must represent productions, with the form LHS -> RHS.  If -> is missing the line is malformed.
        if "->" not in cleaned:
            raise ValueError(
                f"Line {line_number}: production must contain '->'. Found: "
                f"{original_line!r}"
            )

        lhs_text, rhs_text = cleaned.split("->", 1)
        lhs = lhs_text.strip()
        if not lhs:
            raise ValueError(
                f"Line {line_number}: production is missing a left-hand side."
            )

        # Split the right-hand side into individual alternatives using |
        alternatives = [alt.strip() for alt in rhs_text.split("|")]
        if not all(alternatives):
            raise ValueError(
                f"Line {line_number}: found an empty alternative in {original_line!r}"
            )

        rhs_list: List[ProductionRHS] = []
        for alternative in alternatives:
            # alternative may be a terminal such as "cake" or a sequence of
            # symbols separated by spaces, for instance Det N
            symbols = [
                _strip_quotes(token)
                for token in alternative.split()
                if token.strip()
            ]
            if len(symbols) == 1 and symbols[0] in {"E", "e"}:
                rhs_list.append(tuple())
            else:
                rhs_list.append(tuple(symbols))

        # Store the collected alternatives under the left-hand side symbol, if already exists append to the existing def
        productions.setdefault(lhs, []).extend(rhs_list)

        # If the start symbol has not been defined yet, default to the first
        if start_symbol is None:
            start_symbol = lhs

    if start_symbol is None:
        raise ValueError(
            "The grammar file does not contain any productions, so the start "
            "symbol could not be inferred.  Please add at least one rule."
        )

    return Grammar(start=start_symbol, productions=productions)


def tokenize(sentence: str) -> List[str]:
    """
    Tokenise sentence
    """

    # Normalise to lower case to match the terminals defined in the grammar
    lower_sentence = sentence.lower()

    # findall returns only the substrings that match the pattern
    tokens = _TOKEN_RE.findall(lower_sentence)

    return tokens


def _strip_quotes(token: str) -> str:
    """
    Remove surrounding single/double quotes from token.
    """

    return token.strip(" '\"")
