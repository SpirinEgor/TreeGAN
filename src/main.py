import logging

import torch

from src.grammar import Grammar
from src.modules.tree_generator import TreeGenerator
from src.tree import Tree


def main():
    logging.basicConfig(level=logging.INFO)

    grammar = Grammar("grammars/palindrom/palindrom.lark")

    tree_generator = TreeGenerator(grammar, 128, 128)
    lhs, parent_lhs, predicted_rules = tree_generator(torch.rand([1, 128]), max_actions=10)
    tree = Tree.from_lhs_sequence(lhs, parent_lhs, predicted_rules, grammar)
    print(tree)


if __name__ == "__main__":
    main()
