import logging
import symbol
from collections import OrderedDict
from typing import List, Tuple

import torch
from lark import Lark
from lark.grammar import Terminal, Rule

logger = logging.getLogger(__name__)


RULES = List[Tuple[int, List[int]]]


class Grammar:
    def __init__(self, lark_file_path: str, *, lark_start_rule: str = "start"):
        logger.info(f"Use grammar from {lark_file_path}")
        self.lark_file_path = lark_file_path
        with open(lark_file_path, "r") as f:
            lark_grammar = f.read()
        self.lark_parser = Lark(lark_grammar, start=lark_start_rule)
        self.lark_start_rule = lark_start_rule

        self.non_terminal_to_id = OrderedDict()
        for rule in self.lark_parser.rules:
            if rule.origin not in self.non_terminal_to_id:
                self.non_terminal_to_id[rule.origin] = len(self.non_terminal_to_id)

        self.terminal_to_id = OrderedDict(
            [(t.name, i + len(self.non_terminal_to_id)) for i, t in enumerate(self.lark_parser.terminals)]
        )

        self.id_to_name = {v: k.name for k, v in self.non_terminal_to_id.items()}
        self.id_to_name.update({v: k for k, v in self.terminal_to_id.items()})

        logger.info(
            f"Found {len(self.lark_parser.rules)} rules, "
            f"{len(self.terminal_to_id)} terminals, "
            f"{len(self.non_terminal_to_id)} non-terminals"
        )

    @property
    def n_non_terminals(self) -> int:
        return len(self.non_terminal_to_id)

    @property
    def start_id(self) -> int:
        for i, rule in enumerate(self.lark_parser.rules):
            if self.lark_start_rule == rule.origin.name:
                return i

    def is_terminal(self, symbol_id: int) -> bool:
        return symbol_id >= self.n_non_terminals

    def get_symbol_id(self, symbol) -> int:
        if isinstance(symbol, Terminal):
            return self.terminal_to_id[symbol.name]
        else:
            return self.non_terminal_to_id[symbol]

    def get_symbol_name(self, symbol_id: int) -> str:
        return self.id_to_name[symbol_id]

    def build_non_terminal_rule_matrix(self) -> torch.Tensor:
        """Build matrix for mapping non-terminal to possible rules.
        Each row in a matrix represents a non-terminal.
        Each column in a matrix represents a rule.
        M[i, j] -- true or false whether rule j has non-terminal i as a head.

        :return: matrix
        """
        rules = self.lark_parser.rules
        matrix = torch.zeros([len(self.non_terminal_to_id), len(rules)], dtype=torch.bool)
        for i, rule in enumerate(rules):
            matrix[self.non_terminal_to_id[rule.origin], i] = True
        return matrix

    def rules(self) -> RULES:
        rules = []
        for rule in self.lark_parser.rules:
            rules.append((self.non_terminal_to_id[rule.origin], [self.get_symbol_id(it) for it in rule.expansion]))
        return rules
