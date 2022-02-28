import logging
from typing import Optional

import torch
from torch import nn

from src.grammar import Grammar

logger = logging.getLogger(__name__)


class TreeGenerator(nn.Module):
    def __init__(
        self,
        grammar: Grammar,
        emb_dim: int,
        hid_dim: int,
    ):
        super().__init__()
        self.grammar = grammar
        self.rules = grammar.rules()
        self.non_terminal_rules_matrix = nn.Parameter(grammar.build_non_terminal_rule_matrix(), requires_grad=False)

        self.n_non_terminals, n_rules = self.non_terminal_rules_matrix.shape

        self.padding_id = self.n_non_terminals
        self.lhs_embedding = nn.Embedding(self.n_non_terminals + 1, emb_dim, padding_idx=self.padding_id)
        self.lstm = nn.LSTMCell(2 * emb_dim, hid_dim)
        self.rule_linear = nn.Linear(hid_dim, n_rules)

        self.hid_dim = hid_dim

    def forward(self, z: torch.Tensor, max_actions: Optional[int] = None):
        parent_lhs_emb_stack = [(self.padding_id, self.lhs_embedding(torch.tensor(self.padding_id, device=z.device)))]
        current_lhs_stack = [self.grammar.start_id]

        # [1; hid]
        h, c = z, z

        lhs_sequence, parent_lhs_sequence, predicted_rules = [], [], []
        while True:
            if len(current_lhs_stack) == 0:
                break
            if max_actions is not None and len(lhs_sequence) == max_actions:
                break

            parent_lhs, parent_lhs_emb = parent_lhs_emb_stack.pop()
            current_lhs = current_lhs_stack.pop()

            if self.grammar.is_terminal(current_lhs):
                lhs_sequence.append(current_lhs)
                parent_lhs_sequence.append(parent_lhs)
                predicted_rules.append(None)
                continue

            current_lhs_emb = self.lhs_embedding(torch.tensor(current_lhs, device=z.device))

            lhs_sequence.append(current_lhs)
            parent_lhs_sequence.append(parent_lhs)

            # [1; 2 * emb]
            rnn_input = torch.cat([parent_lhs_emb, current_lhs_emb], dim=0).unsqueeze(0)
            # [1; hid]
            h, c = self.lstm(rnn_input, (h, c))

            # [1; n rules]
            rule_logits = torch.log_softmax(self.rule_linear(h), dim=-1)
            rule_logits[0, ~self.non_terminal_rules_matrix[current_lhs]] = -1e9

            # [1]
            predicted_rule = rule_logits.argmax(dim=1)
            predicted_lhs = self.rules[predicted_rule][0]
            assert predicted_lhs == current_lhs
            predicted_rules.append(predicted_rule.item())

            for predicted_rhs in reversed(self.rules[predicted_rule][1]):
                parent_lhs_emb_stack.append((current_lhs, current_lhs_emb))
                current_lhs_stack.append(predicted_rhs)

        return lhs_sequence, parent_lhs_sequence, predicted_rules
