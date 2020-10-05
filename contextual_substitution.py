from collections import namedtuple
from typing import List

import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast

Substitutions = namedtuple('Substitutions', ('token', 'probs', 'substitutions'))


class Model:
    def __init__(self, model_name, cache_dir=None) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    def generate_substitutions(
            self, string: str, n: int = 10, double_context: bool = False) -> List[Substitutions]:
        token_ids = self.tokenizer(string)['input_ids']

        if double_context:
            mask_bound = len(token_ids)
            token_ids.extend(token_ids[1:])
        else:
            mask_bound = 1

        masked_tokens = None
        masked_positions = range(mask_bound, len(token_ids) - 1)

        for i in masked_positions:
            token_id = token_ids[i]
            if token_id not in self.tokenizer.all_special_ids:
                token_ids[i] = self.tokenizer.mask_token_id
                if masked_tokens is None:
                    masked_tokens = torch.LongTensor(token_ids).unsqueeze(0)
                else:
                    masked_tokens = torch.cat(
                        (masked_tokens, torch.LongTensor(token_ids).unsqueeze(0)), dim=0)
                token_ids[i] = token_id

        masked_tokens = masked_tokens.to(self.device)
        with torch.no_grad():
            logits = self.model(masked_tokens)[0]

        result = []

        for i, masked_position in enumerate(masked_positions):
            probs, indices = torch.topk(logits[i, masked_position].softmax(dim=0), n)
            token = self.tokenizer.convert_ids_to_tokens(token_ids[masked_position])
            probs = probs.tolist()
            substitutions = self.tokenizer.convert_ids_to_tokens(indices.tolist())
            result.append(Substitutions(token, probs, substitutions))

        return result
