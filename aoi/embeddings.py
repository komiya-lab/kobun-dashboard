import os
import torch
from aoi.config import AoiConfig
from torch import Tensor
from typing import Any, List, Dict
from registrable import Registrable
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Embedding(Registrable):

    def __init__(self, config: AoiConfig) -> None:
        self.config = config
        super().__init__()
    
    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return self.encode(*args, **kwds)
    
    def encode(self, *args: Any, **kwds: Any) ->Tensor:
        raise NotImplementedError
    

@Embedding.register("T5-kokub-translation")
class T5KobunTranslationEmbedding(Embedding):

    def __init__(self, config: AoiConfig) -> None:
        super().__init__(config)
        self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        model_path = os.path.join(config.modeldir, "notitle_learningrate_0.0002_epoch_10_penalty_1.0_data_0/T5_koten_model/")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def encode(self, tokens: List[str], index: int) -> Tensor:
        input_ids = list()
        begin = -1
        end = -1
        for i, token in enumerate(tokens):
            ids = self.tokenizer(token, return_tensors="pt").input_ids
            if index == i:
                begin = sum([len(t[0]) for t in input_ids])
                if begin == 0:
                    begin = 1
                if ids[0, 0].item() != 5: # 5はbegin token
                    end = begin + len(ids[0]) -1
                else:
                    end = begin + len(ids[0]) -2

            if self.config.debug and token == "初冠":
                print(ids)
            if len(input_ids) == 0:
                input_ids.append(ids[:, 0:-1])
            elif i == len(tokens) -1:
                input_ids.append(ids[:, 1:])
            elif ids[0, 0].item() != 5: # 5はbegin token
                input_ids.append(ids[:, 0:-1])
            else:
                input_ids.append(ids[:, 1:-1])
            
        inp = torch.concat(input_ids, dim=1)
        if self.config.debug:
            print(self.tokenizer.convert_ids_to_tokens(inp[0]))
            print(begin, end)
        with torch.no_grad():
            out = self.model.encoder(inp).last_hidden_state # (1, len(subword tokens), 768)
            embs = out[0, begin:end, :]

            if self.config.emb_pooling == "sum":
                emb = torch.sum(embs, dim=0)
                return emb
            else:
                emb = torch.mean(embs, dim=0)
                return emb


if __name__ == "__main__":
    emb = T5KobunTranslationEmbedding(AoiConfig("../data/models/", "", "", "mean", True))
    tokens = ["むかし", "、", "男", "、", "初冠", "し", "て", "、", "奈良", "の", "京", "春日", "の", "里", "に", "、", "しる", "よし", 
              "し", "て", "、", "狩", "に", "いに", "けり", "。"]
    print(emb.encode(tokens, 0))