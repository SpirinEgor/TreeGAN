from typing import List

from src.grammar import Grammar


class Tree:
    def __init__(self, grammar: Grammar, nodes_types: List[int], parents: List[int]):
        self.grammar = grammar
        self.nodes_types = nodes_types
        self.parents = parents

    @classmethod
    def from_lhs_sequence(cls, lhs: List[int], parent_lhs: List[int], rule_id: List[int], grammar: Grammar) -> "Tree":
        rules = grammar.rules()

        node_types = [lhs[0]]
        parents = [-1]

        dfs_stack = [(0, lhs[0])]
        for cur_lhs, p_true_lhs, cur_rule_id in zip(lhs[1:], parent_lhs[1:], rule_id[1:]):
            p_node_id, p_lhs = dfs_stack.pop()
            assert p_lhs == p_true_lhs

            cur_node_id = len(node_types)
            node_types.append(cur_lhs)
            parents.append(p_node_id)

            if grammar.is_terminal(cur_lhs):
                continue

            assert rules[cur_rule_id][0] == cur_lhs
            for _ in rules[cur_rule_id][1]:
                dfs_stack.append((cur_node_id, cur_lhs))

        return cls(grammar, node_types, parents)

    def __repr__(self) -> str:
        node_depths = {0: 0}
        nodes_str = [self.grammar.get_symbol_name(self.nodes_types[0])]

        for cur_id, (node_type, parent_id) in enumerate(zip(self.nodes_types[1:], self.parents[1:])):
            depth = node_depths[parent_id] + 1
            nodes_str.append("." * depth + " " + self.grammar.get_symbol_name(node_type))
            node_depths[cur_id + 1] = depth

        return "\n".join(nodes_str)
