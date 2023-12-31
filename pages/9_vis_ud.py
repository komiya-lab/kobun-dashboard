import os
import glob
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from spacy import displacy
from conllu import parse_incr
from collections import Counter

DBPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db/visud")
UPOS_TARGETS = ['PRON_NOUN',
 'VERB_AUX',
 'NOUN_VERB',
 'CCONJ_SCONJ',
 'ADV_NOUN',
 'AUX_ADP',
 'PART_NOUN',
 'ADJ_NOUN',
 'AUX_VERB',
 'NOUN_ADJ',
 'ADP_SCONJ',
 'NOUN_PROPN',
 'NOUN_PUNCT',
 'CCONJ_ADV',
 'VERB_NOUN',
 'CCONJ_NOUN',
 'ADP_PART',
 'VERB_ADV',
 'NOUN_ADV',
 'NOUN_PART']

REL_TARGETS = [
 'nmod',
 'aux',
 'compound',
 'cc',
 'ccomp',
 'cop',
 'obl',
 'case',
 'iobj',
 'advcl',
 'advmod',
 'acl',
 'appos',
 'dep',
 'amod',
 'nsubj',
 "csubj",
 'root',
 'reparandum',
 'obj',
 'mark',
 'nummod',
 'punct',
 'det',
 'discourse',
 'dislocated',
 'fixed'
 ]

d = {
  "words": [
    { "text": "This", "tag": "DT" },
    { "text": "is", "tag": "VBZ" },
    { "text": "a", "tag": "DT" },
    { "text": "sentence", "tag": "NN" }
  ],
  "arcs": [
    { "start": 0, "end": 1, "label": "nsubj", "dir": "left" },
    { "start": 2, "end": 3, "label": "det", "dir": "left" },
    { "start": 1, "end": 3, "label": "attr", "dir": "right" }
  ]
}
ptree = displacy.render(d, style="dep", manual=True, options={"compact": True})
st.set_page_config(layout="wide")

def tokenlist2sentence(tlist) -> str:
    sent = ''.join([t["form"] for t in tlist])
    return sent

def load_sentences(conllufile: str):
    with open(conllufile) as f:
        return {tokenlist2sentence(t): t for t in parse_incr(f)}

def compare_tag(tag, gold, pred, diff, count_gold=False):
    if count_gold:
        for gtoken in gold:
            gtag = gtoken[tag]
            l = diff.get(gtag, list())
            l.append(gtoken)
            diff[gtag] = l
        return diff
    
    for gtoken, ptoken in zip(gold, pred):
        gtag = gtoken[tag]
        ptag = ptoken[tag]
        if gtag == ptag:
            continue
        l = diff.get((gtag, ptag), list())
        l.append([gtoken, ptoken])
        diff[(gtag, ptag)] = l
    return diff

def corpus_stats(corpus_, target="deprel"):
   l = list()
   for v in corpus_.values():
      l.extend([t[target] for t in v])
   L = len(l)
   c = Counter(l)
   val = sorted(c.items(), key=lambda v: v[1], reverse=True)
<<<<<<< HEAD
   return pd.DataFrame([{"type":k, "count": v} for k, v in val])

def normalize(stats, ref):
  if isinstance(ref, (int, float)):
     ret
=======
   return pd.DataFrame([{"count": v, "ratio": v/L*100.0} for _, v in val], index=[k for k,_ in val])
      
>>>>>>> 742b8b6feba37f082f6458698b8e5e563c23790c

def to_dataset(gold, pred, upos_targets, rel_targets):
    res = []
    pos_stats = dict()
    rel_stats = dict()
    for sent, glist in gold.items():
      r = {"sentence": sent, "UPOS": [], "REL": []}
      r.update({t: None for t in upos_targets})
      if sent not in pred:
         continue
      plist = pred[sent]

      gtriples = [(glist[t["head"]-1]["form"], t["form"], t["deprel"]) for t in glist]
      ptriples = [(plist[t["head"]-1]["form"], t["form"], t["deprel"]) for t in plist]
      #ptriples = [(t["head"], t["id"], t["deprel"]) for t in plist]

      for gtoken, ptoken in zip(glist, plist):
        gtag = gtoken["upos"]
        ptag = ptoken["upos"]
        key = f"{gtag}_{ptag}"
        if key in upos_targets and key not in r["UPOS"]:
          r["UPOS"].append(key)
          r[key] = gtoken["lemma"]
        if gtag != ptag:
           pos_stats[key] = pos_stats.get(key, 0) + 1
        
      for triple in gtriples:
        if triple not in ptriples:
          rel = triple[2]
          if rel in rel_targets and rel not in r["REL"]:
            r["REL"].append(rel)
          rel_stats[rel] = rel_stats.get(rel, 0) + 1
      
      if len(r["REL"]) == 0 and len(r["UPOS"]) == 0:
         continue
      res.append(r)
    #print(res)
    return (pd.DataFrame(res), 
      pd.DataFrame([{"count": v} for k, v in pos_stats.items()], index=[k for k,_ in pos_stats.items()]), 
      pd.DataFrame([{"count": v} for k, v in rel_stats.items()], index=[k for k,_ in rel_stats.items()]))
      

def dep_distance(gold, pred):
    distances = dict()   

    for sent, glist in gold.items():
      r = {"sentence": sent, "UPOS": [], "REL": []}
      r.update({t: None for t in upos_targets})
      if sent not in pred:
         continue
      plist = pred[sent]

      gtriples = [(glist[t["head"]-1]["form"], t["form"]) for t in glist]
      ptriples = [(plist[t["head"]-1]["form"], t["form"]) for t in plist]
      #gtriples = [(t["head"], t["id"]) for t in glist]
      #ptriples = [(t["head"], t["id"]) for t in plist]
        
      for i, triple in enumerate(gtriples):
        dist = int(np.abs(glist[i]["head"]-glist[i]["id"]))
        r = distances.get(dist, {"gold": 0, "correct": 0, "error": 0, "pred": 0})
        r["gold"] += 1
        if triple not in ptriples:
          r["error"] += 1
        else:
           r["correct"] += 1
        distances[dist] = r
      
      for i, triple in enumerate(ptriples):
        dist = int(np.abs(plist[i]["head"]-plist[i]["id"]))
        r = distances.get(dist, {"gold": 0, "correct": 0, "error": 0, "pred": 0})
        r["pred"] += 1
        distances[dist] = r

    #print(res)
    res = list()
    for k, v in distances.items():
      for kk, vv in v.items():
         res.append({"distance": k, "count": vv, "type": kk})
    df1 = pd.DataFrame.from_dict(distances, orient="index").sort_index(axis=0)
    df1["error_rate"] = df1["error"].div(df1["gold"])
    df2 = pd.DataFrame(res).sort_values("distance")
    print(df1)
    return df1, df2

    
def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def dep_render(tlist):
  d = {
    "words": [{"text": t["form"], "tag": t["upos"]} for t in tlist],
    "arcs": []
  }
  for t in tlist:
    if t["head"] == 0:
      continue
    if t["head"] > t["id"]:
      d["arcs"].append({"start": t["id"]-1, "end": t["head"]-1, "label": t["deprel"], "dir": "left"})
    else:
      d["arcs"].append({"start": t["head"]-1, "end": t["id"]-1, "label": t["deprel"], "dir": "right"})
  return displacy.render(d, style="dep", manual=True, options={"compact": True})

def plot_stats(stats: pd.DataFrame, target="count", normalize=None):
   if normalize is not None:
     target = "normalized"
     stats = stats.copy()
     print(normalize["count"])
     stats[target] = stats["count"].div(normalize["count"])
   print(stats)
   print(normalize)
   stats = stats.sort_values(by=target, ascending=False)
   return px.bar(stats, x=stats.index, y=target)


corpora = ("ja_modern-ud", "maihime", "yukiguni", "koyayori", "ja_bccwj-ud", "ja_gsd-ud")
parsers = list()
for c in corpora:
  parsers.extend([os.path.basename(p) for p in glob.glob(os.path.join(DBPATH, f"{c}/*"))])
parsers = sorted(set(parsers))

with st.sidebar:
    corpus  = st.selectbox("corpus", corpora)
    gold = st.selectbox("gold", ["gold"] + parsers)
    parser = st.selectbox("parser", parsers)

if gold == "gold":
  gold_file = os.path.join(DBPATH, f"{corpus}/gold.conllu")
else:
  gold_file = os.path.join(DBPATH, f"{corpus}/{gold}/pred.conllu")
pred_file = os.path.join(DBPATH, f"{corpus}/{parser}/pred.conllu")

gold_sentences = load_sentences(gold_file)
pred_sentences = load_sentences(pred_file)

with st.sidebar:
  upos_targets = st.multiselect("upos targets", UPOS_TARGETS, default=UPOS_TARGETS[:3])
  rel_targets = st.multiselect("rel targets", REL_TARGETS, default=REL_TARGETS)

gold_stats = corpus_stats(gold_sentences)
pred_stats = corpus_stats(pred_sentences)
df, pos_stats, rel_stats = to_dataset(gold_sentences, pred_sentences, upos_targets, rel_targets)
selected = dataframe_with_selections(df)

st.markdown("## Corpora Stats")
col1, col2 = st.columns(2)

with col1:
  st.markdown(f"## {gold} stats \n ### count")
  st.plotly_chart(plot_stats(gold_stats))
  st.markdown("### ratio")
  st.plotly_chart(plot_stats(gold_stats, "ratio"))

with col2:
  st.markdown(f"## {parser} stats \n ### count")
  st.plotly_chart(plot_stats(pred_stats))
  st.markdown("### ratio")
  st.plotly_chart(plot_stats(pred_stats, "ratio"))

st.markdown("## Rel Distances")
distances, plot_dist = dep_distance(gold_sentences, pred_sentences)
st.download_button("Download distances CSV", data=distances.to_csv().encode('utf-8'),
    file_name=f'{corpus}_{parser}.distances.csv',
    mime='text/csv')

col1, col2 = st.columns(2)
with col1:
  fig = px.line(plot_dist, x="distance", y="count", color="type", log_y=True)
  st.plotly_chart(fig)

with col2:
   st.plotly_chart(px.line(distances, x=distances.index, y="error_rate"))

st.markdown("## Differences between corpora")
d_data = gold_stats.copy()
d_data["rel_count"] = rel_stats["count"]
st.download_button("Download rel stats", data=d_data.to_csv().encode('utf-8'),
    file_name=f'{corpus}_{parser}.relstats.csv',
    mime='text/csv',)

col1, col2 = st.columns(2)

with col1:
  if len(pos_stats) == 0:
    st.write("no pos difference")
  else:
    st.plotly_chart(plot_stats(pos_stats))
    st.markdown("### normalized by gold")
    st.plotly_chart(plot_stats(rel_stats, normalize=gold_stats))

with col2:
  if len(rel_stats) == 0:
    st.write("no pos difference")
  else:
    st.plotly_chart(plot_stats(rel_stats))
    st.markdown("### normalized by pred")
    st.plotly_chart(plot_stats(rel_stats, normalize=pred_stats))

upt = st.selectbox("plot upos target", upos_targets)
up_count = df[upt].value_counts(ascending=False)

col1, col2 = st.columns([0.2, 0.8])
with col1:
  st.dataframe(up_count)

with col2:
  fig = px.bar(up_count)
  st.plotly_chart(fig)

if len(selected) > 0:
  st.markdown(f"## Gold: {gold}")
  gold_dep = dep_render(gold_sentences[selected["sentence"].values[0]])
  st.download_button("Download gold SVG",  data=gold_dep,
    file_name=f'{corpus}_{parser}.gold.svg',
    mime='image/svg+xml')
  st.write(gold_dep, unsafe_allow_html=True)

  st.markdown(f"## Pred: {parser}")
  pred_dep = dep_render(pred_sentences[selected["sentence"].values[0]])
  st.download_button("Download gold SVG",  data=pred_dep,
    file_name=f'{corpus}_{parser}.pred.svg',
    mime='image/svg+xml')
  st.write(pred_dep, unsafe_allow_html=True)