import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from typing import List
import plotly.express as px
from aoi.config import AoiConfig
from aoi.dataset import CHJDataset
from aoi.embeddings import Embedding
from sklearn.mixture import GaussianMixture

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PJDIR = os.path.join(BASEDIR, "projects")
CONFIGFILE = os.path.join(BASEDIR, "config.json")

with open(CONFIGFILE) as f:
    js = json.load(f)
    CONFIG = AoiConfig(**js)

CHJ_DATADIR = os.path.join(CONFIG.dbdir, "chj3")
CHJ_DATA = CHJDataset(CHJ_DATADIR)

#SURFACE_POS_INV = dict()
POSS = CHJ_DATA.get_pos("")
POSS.sort()
TARGETS = ["サブコーパス名" , "ジャンル", "作品名"]

def count(counts: dict, data: dict, keys: List[str]=TARGETS):
    for key in keys:
        d = counts.get(key, dict())
        d[data[key]] = d.get(data[key], 0) + 1
        counts[key] = d
    return counts

prj = st.selectbox("project", [os.path.basename(name) for name in glob.glob(os.path.join(PJDIR, "./*"))])

with st.form("setup"):
    surface = st.text_input("語彙素")
    pos = st.selectbox("品詞", POSS)
    emb_class = st.selectbox("埋込表現", Embedding.list_available())
    min_k = st.slider("最小クラスタ数", 2, 10, 3, 1)
    max_k = st.slider("最大クラスタ数", 3, 50, 10, 1)
    sample_num = st.number_input("最大サンプル数", -1, 100000, 10000)
    sbm = st.form_submit_button("分析")


L = CHJ_DATA.len(surface, pos) if sample_num < 0 else min(CHJ_DATA.len(surface, pos), sample_num)
st.write(f"全 {L} サンプル")
analysis_bar = st.progress(0.0, "分析進捗")
outdir = os.path.join(PJDIR, prj, "embs")
calculated = False

if sbm:
    emb = Embedding.by_name(emb_class)(CONFIG)
    embs = dict()

    if not CHJ_DATA.exists(surface, pos):
        st.toast(f"語彙素: {surface}, 品詞: {pos} は存在しません。")
        analysis_bar.progress(1.0, f"語彙素: {surface}, 品詞: {pos} は存在しません。")
    else:
        # extract embs
        counts = dict()
        indices = list()
        for i, (index, data) in enumerate(CHJ_DATA.sample(surface, pos, sample_num)):
            print(i/L)
            analysis_bar.progress(np.min((i/L, 1.0))*0.5)
            if "tokens" not in data:
                continue
            count(counts, data)
            tokens = list()
            target = -1
            for j, token in enumerate(data["tokens"]):
                if token["語彙素"] == surface and token["品詞"] == pos:
                    target = j
                tokens.append(token["書字形出現形"])
            e = emb.encode(tokens, target)
            embs[index] = e.tolist()
            indices.append(index)


        # gmm
        gmms = list()
        features = np.array(list(embs.values()))
        aucs = list()
        for k in range(min_k, max_k + 1):
            print(k)
            analysis_bar.progress(0.5 + k/(max_k - min_k) * 0.5, "GMM clustering")
            gmm = GaussianMixture(k)
            gmm.fit(features)
            gmms.append(gmm)
            aucs.append(gmm.aic(features))
        
        auc_i = int(np.argmin(aucs))
        print(auc_i)
        best_cluster_k  = min_k + auc_i
        results = {"best_cluster_k": min_k + auc_i, "auc": aucs[auc_i], "surface": surface, "pos": pos, "counts": counts, "aucs": aucs,
                   "indices": indices, "min_k": min_k, "max_k": max_k, "emb": emb_class}
        best_cluster = gmms[auc_i]

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False)

        with open(os.path.join(outdir, "best_cluster.pkl"), "wb") as f:
            pickle.dump(best_cluster, f)

        with open(os.path.join(outdir, "embs.pkl"), "wb") as f:
            pickle.dump(embs, f)

        calculated = True

if os.path.exists(outdir):
    if not calculated:
        with open(os.path.join(outdir, "results.json")) as f:
            results = json.load(f)
            counts = results["counts"]
            surface = results["surface"]
            indices = results["indices"]
            best_cluster_k = results["best_cluster_k"]
            pos = results["pos"]
            tpl = (surface, pos)

        with open(os.path.join(outdir, "best_cluster.pkl"), "rb") as f:
            best_cluster = pickle.load(f)

        with open(os.path.join(outdir, "embs.pkl"), "rb") as f:
            embs = pickle.load(f)

    st.write(f"## {surface} {pos} クラスタ数: {best_cluster_k}")
    panes = st.columns(len(counts))

    for pane, key in zip(panes, counts):
        data = sorted(counts[key].items(), key=lambda v: v[1], reverse=True)
        with pane:
            fig = px.bar(x=[d[0] for d in data], y=[d[1] for d in data], title=key)
            st.plotly_chart(fig)

    data = list()
    targets = TARGETS + ["成立年", "cluster", "count", "sentence"]
    for index in indices:
        d = CHJ_DATA.get(index)
        e = embs[index]
        cls = best_cluster.predict([e])
        d["cluster"] = int(cls[0])
        d["count"] = 1
        data.append([d[key] for key in targets])

    data = pd.DataFrame(data, columns=targets)
    data = data.sort_values("cluster")

    panes = st.columns(len(targets) -3)

    for pane, key in zip(panes, targets[:-3]):
        with pane:
            fig = px.bar(data, x=key, y="count", color="cluster", title=key)
            st.plotly_chart(fig)

    st.dataframe(data)
