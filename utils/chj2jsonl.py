import typer
import csv
import json
import pickle
import uuid
import hashlib
import os
from typing import Tuple, List

app = typer.Typer()

def load_chj(fname: str):
    # generatorにすることでメモリを節約
    with open(fname) as f:
        reader = csv.reader(f, delimiter="\t")
        head = next(reader)
        for row in reader:
            yield dict(zip(head, row))

def to_sentences(data):
    targets = [
 '開始位置',
 '連番',
 'コア',
 '書字形出現形',
 '語彙素 ID',
 '語彙素読み',
 '語彙素',
 '語彙素細分類',
 '語形',
 '語形代表表記',
 '品詞',
 '活用型',
 '活用形',
 '書字形',
 '仮名形出現形',
 '発音形出現形',
 '語種',
 '原文文字列',
 '振り仮名']
    meta = ['サブコーパス名',
 'サンプル ID',
 '本文種別',
 '話者',
 '文体',
 '歌番号',
 'ジャンル',
 '作品名',
 '成立年',
 '巻名等',
 '部',
 '作者',
 '生年',
 '性別',
 '底本',
 'ページ番号',
 '校注者',
 '出版社']
    buf_a = list()
    buf_n = list() 
    buf_t = list()
    prv_id = None
    for d in data:
        a_ = d["原文文字列"]
        n_ = d["書字形出現形"]
        id_ = d["サンプル ID"]
        if prv_id is None:
            prv_id = id_
            
        if prv_id != id_ and len(buf_a) > 0:
            dd = {"sentence": "".join(buf_a), "sentence_normalize": "".join(buf_n), "tokens": buf_t}
            dd.update({k: d[k] for k in meta})
            yield dd
            buf_a.clear()
            buf_n.clear()
            buf_t.clear()
        
        prv_id = id_
        buf_a.append(a_)
        buf_n.append(n_)
        buf_t.append({k: d[k] for k in targets})
        
        if a_ == "。":
            dd = {"sentence": "".join(buf_a), "sentence_normalize": "".join(buf_n), "tokens": buf_t}
            dd.update({k: d[k] for k in meta})
            yield dd
            buf_a.clear()
            buf_n.clear()
            buf_t.clear()
            
    if len(buf_a) > 0:
        dd = {"sentence": "".join(buf_a), "sentence_normalize": "".join(buf_n), "tokens": buf_t}
        dd.update({k: d[k] for k in meta})
        yield dd
            

@app.command()
def chj2jsonl(fname: str):
    for d in to_sentences(load_chj(fname)):
        print(json.dumps(d, ensure_ascii=False))


@app.command()
def db(jsonlname: str, outdir: str, keys: List[str]=("語彙素", "品詞")):
    datadir = os.path.join(outdir, "./data")
    indicesdir = os.path.join(outdir, "./indices")
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        os.makedirs(indicesdir)

    indices = dict()
    with open(jsonlname) as f:
        c = 0
        for line in f:
            c += 1
            js = json.loads(line)
            id_ = hashlib.md5(line.encode()).hexdigest()
            if "tokens" not in js:
                with open(os.path.join(datadir, id_), 'w') as f2:
                    f2.write(line)
                continue

            if c % 1000 == 0:
                print(f"extracted {c} sentences.")
            for token in js["tokens"]:
                vals = tuple((token[k] for k in keys))
                #print(vals)
                l = indices.get(vals, list())
                l.append(id_)
                indices[vals] = l

            if not os.path.exists(os.path.join(datadir, id_)):
                with open(os.path.join(datadir, id_), 'w') as f2:
                    f2.write(line)
    
    c = 0
    for k, v in indices.items():
        c += 1
        v = list(set(v))
        indices[k] = v
        if c % 1000 == 0:
            print(f"extracted {c} lexicons.")
        #print(k, v)
        with open(os.path.join(indicesdir, "_".join(k)), 'w') as f:
            json.dump(v, f, ensure_ascii=False)

    with open(os.path.join(outdir, "indices.json"), "w") as f:
        json.dump(indices, f, ensure_ascii=False)


if __name__ == "__main__":
    app()