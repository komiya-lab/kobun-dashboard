import os
import glob
import json
import numpy as np
from typing import Any, List, Dict, Iterable, Tuple

class CHJDataset:

    def __init__(self, location) -> None:
        self.location = location
        self.datapath = os.path.join(location, "data")
        self.indices = {os.path.basename(fname): fname for fname in glob.glob(os.path.join(location, "./indices/*"))}
        self.surface2pos = dict()
        self.pos2surface = dict()
        for k in self.indices:
            s, p = k.split("_")[:2]
            l = self.surface2pos.get(s, list())
            l.append(p)
            self.surface2pos[s] = l

            l = self.pos2surface.get(p, list())
            l.append(s)
            self.pos2surface[p] = l

    def get(self, id_:str, default:Any = None) -> Dict:
        fpath = os.path.join(self.datapath, id_)
        if not os.path.exists(fpath):
            return default
        with open(fpath) as f:
            return json.load(f)
        
    def itr_find(self, surface: str, pos: str) -> Iterable[Tuple[str, Dict]]:
        key = f"{surface}_{pos}"
        if key in self.indices:
            with open(self.indices[key]) as f:
                indices = json.load(f)
            for ind in indices:
                yield ind, self.get(ind)

    def sample(self, surface: str, pos: str, max_sample: int) -> List[Tuple[str, Dict]]:
        key = f"{surface}_{pos}"
        if not key in self.indices:
            return "", list()
        
        if max_sample < 0:
            return self.itr_find(surface, pos)

        with open(self.indices[key]) as f:
            indices = json.load(f)
        
        if len(indices) > max_sample:
            np.random.shuffle(indices)
            indices = indices[:max_sample]

        return [(ind, self.get(ind)) for ind in indices]


    def get_surfaces(self, pos: str) -> List[str]:
        return self.pos2surface.get(pos, list())
    
    def get_pos(self, surface: str) -> List[str]:
        if surface == "":
            return list(self.pos2surface.keys())
        
        return self.surface2pos.get(surface, list())
    
    def exists(self, surface: str, pos: str) -> bool: 
        key = f"{surface}_{pos}"
        return key in self.indices
    
    def len(self, surface: str, pos: str) -> int: 
        key = f"{surface}_{pos}"
        if key not in self.indices:
            return 0
        
        with open(self.indices[key]) as f:
            indices = json.load(f)
        return len(indices)


if __name__ == "__main__":
    dataset = CHJDataset("db/chj3")
    print(dataset.get_pos(""))