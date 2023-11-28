from dataclasses import dataclass


@dataclass
class AoiConfig:
    modeldir: str
    corpusdir: str
    dbdir: str
    emb_pooling: str
    debug: bool = False