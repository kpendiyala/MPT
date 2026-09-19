#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def entropy_counts(x):
    x=np.asarray(x,float); s=x.sum()
    if s<=0: return 0.0
    p=x[x>0]/s
    return float(-(p*np.log(p)).sum())

def nmi(x,y):
    tab=pd.crosstab(pd.Series(x),pd.Series(y))
    n=tab.values.sum()
    if n==0: return np.nan
    pxy=tab.values/n; px=pxy.sum(1,keepdims=True); py=pxy.sum(0,keepdims=True)
    nz=pxy>0
    mi=float((pxy[nz]*np.log(pxy[nz]/(px@py)[nz])).sum())
    hx=entropy_counts(tab.sum(1).values); hy=entropy_counts(tab.sum(0).values)
    return 0.0 if hx<=0 or hy<=0 else mi/math.sqrt(hx*hy)

def js(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float)
    p=p/max(p.sum(),1e-12); q=q/max(q.sum(),1e-12); m=.5*(p+q)
    def kl(a,b):
        z=a>0
        return float((a[z]*np.log(a[z]/np.clip(b[z],1e-12,None))).sum())
    return .5*kl(p,m)+.5*kl(q,m)

def truth_col(pred):
    for c in ("truth_label","jet_label","label","truth"):
        if c in pred.columns: return c
    cs=[c for c in pred.columns if "truth" in c.lower() and "score" not in c.lower()]
    return cs[0] if len(cs)==1 else None

def attach_labels(df,pred_path):
    if not pred_path or not Path(pred_path).exists(): return df,None
    p=pd.read_parquet(pred_path); c=truth_col(p)
    if c is None:
        print("WARNING: no scalar truth-label column found; jet-class analyses skipped")
        return df,None
    lab=pd.to_numeric(p[c],errors="coerce")
    if lab.isna().any():
        print(f"WARNING: {c} not scalar numeric; jet-class analyses skipped")
        return df,None
    if len(df) and int(df.event_index.max())>=len(lab):
        raise RuntimeError("Prediction row count does not match collector event indexing")
    out=df.copy()
    out["jet_label"]=lab.iloc[out.event_index.astype(int).to_numpy()].to_numpy(dtype=int)
    out["jet_family"]=np.select(
        [out.jet_label.between(0,14),out.jet_label.between(15,160),out.jet_label.between(161,187)],
        ["Res2P","Res34P","QCD"], default="unknown")
    return out,c

def add_bins(df):
    d=df.copy()
    if "part_logptrel" in d:
        d["logptrel_bin"]=pd.cut(d.part_logptrel,[-np.inf,-6,-5,-4,-3,-2,-1,np.inf],include_lowest=True)
    if "part_deltaR" in d:
        d["deltaR_bin"]=pd.cut(d.part_deltaR,[-np.inf,.02,.05,.1,.2,.4,.8,np.inf],include_lowest=True)
    if "pt_rank" in d:
        d["pt_rank_bin"]=pd.cut(d.pt_rank,[0,1,2,4,8,16,32,64,128],
            labels=["1","2","3-4","5-8","9-16","17-32","33-64","65-128"],include_lowest=True)
    return d

def explode(df,accepted_only=True):
    slots=sorted(int(c.split("_")[1]) for c in df if c.startswith("expert_"))
    out=[]
    base=[c for c in df if not (c.startswith("expert_") or c.startswith("weight_") or c.startswith("accepted_"))]
    for s in slots:
        q=df[base].copy()
        q["slot"]=s; q["expert"]=df[f"expert_{s}"].astype(int)
        q["routing_weight"]=df[f"weight_{s}"].astype(float)
        q["accepted"]=df[f"accepted_{s}"].astype(bool)
        if accepted_only: q=q[q.accepted]
        out.append(q)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def lift_table(a,var):
    rows=[]
    for layer,g in a.groupby("layer",sort=False):
        overall=g.expert.value_counts(normalize=True)
        for group,gg in g.groupby(var,dropna=False):
            cond=gg.expert.value_counts(normalize=True)
            for e in sorted(g.expert.unique()):
                base=float(overall.get(e,0)); frac=float(cond.get(e,0))
                rows.append(dict(layer=layer,layer_index=int(g.layer_index.iloc[0]),
                    variable=var,group=str(group),expert=int(e),fraction=frac,
                    global_fraction=base,lift=(frac/base if base else np.nan),
                    n_assignments=len(gg)))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",required=True)
    ap.add_argument("--predictions")
    args=ap.parse_args()
    root=Path(args.input_dir); out=root/"analysis"; out.mkdir(exist_ok=True)

    tok=add_bins(pd.read_parquet(root/"token_sample.parquet"))
    tok,label_name=attach_labels(tok,args.predictions)
    assign=explode(tok,True)
    usage=pd.read_csv(root/"expert_usage.csv")

    lifts=[]
    for v in ("particle_type","logptrel_bin","deltaR_bin","pt_rank_bin","jet_family"):
        if v in assign: lifts.append(lift_table(assign,v))
    lift=pd.concat(lifts,ignore_index=True) if lifts else pd.DataFrame()
    if not lift.empty: lift.to_csv(out/"conditional_expert_lift.csv",index=False)

    rows=[]
    for layer,g in tok.groupby("layer",sort=False):
        for v in ("particle_type","logptrel_bin","deltaR_bin","pt_rank_bin","jet_family"):
            if v in g:
                m=g[v].notna()
                if m.sum()>=100:
                    rows.append(dict(layer=layer,layer_index=int(g.layer_index.iloc[0]),
                        variable=v,n_tokens=int(m.sum()),
                        nmi_primary_expert=nmi(g.loc[m,"expert_0"].astype(str),g.loc[m,v].astype(str))))
    nmis=pd.DataFrame(rows)
    nmis.to_csv(out/"routing_nmi.csv",index=False)

    jsrows=[]
    for layer,g in assign.groupby("layer",sort=False):
        for feat,bins in {"part_logptrel":np.linspace(-8,0,33),"part_deltaR":np.linspace(0,1.2,33)}.items():
            if feat not in g: continue
            ex=sorted(g.expert.unique()); vals=[]; best=(-1,None,None)
            hist={e:np.histogram(g.loc[g.expert==e,feat].dropna(),bins=bins)[0] for e in ex}
            for i,a in enumerate(ex):
                for b in ex[i+1:]:
                    d=js(hist[a],hist[b]); vals.append(d)
                    if d>best[0]: best=(d,a,b)
            jsrows.append(dict(layer=layer,layer_index=int(g.layer_index.iloc[0]),feature=feat,
                mean_pairwise_js=(float(np.mean(vals)) if vals else 0),
                max_pairwise_js=(best[0] if vals else 0),max_js_expert_a=best[1],max_js_expert_b=best[2]))
    jsdf=pd.DataFrame(jsrows); jsdf.to_csv(out/"expert_feature_js_divergence.csv",index=False)

    clsrows=[]
    cp=root/"class_token_sample.parquet"
    if cp.exists():
        cls,_=attach_labels(pd.read_parquet(cp),args.predictions)
        if "jet_family" in cls:
            for layer,g in cls.groupby("layer",sort=False):
                clsrows.append(dict(layer=layer,layer_index=int(g.layer_index.iloc[0]),n_events=len(g),
                    nmi_expert_vs_jet_family=nmi(g.expert_0.astype(str),g.jet_family.astype(str)),
                    nmi_expert_vs_exact_class=nmi(g.expert_0.astype(str),g.jet_label.astype(str))))
    pd.DataFrame(clsrows).to_csv(out/"class_token_routing_nmi.csv",index=False)

    # Plot 1: full-data accepted expert utilization by layer.
    piv=usage.pivot(index="layer_index",columns="expert",values="valid_accepted_fraction")
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for e in piv.columns: ax.plot(piv.index,piv[e],marker="o",label=f"Expert {e}")
    ax.set(xlabel="MoE layer index",ylabel="Accepted valid-assignment fraction",title="Layer-wise expert utilization")
    ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(out/"expert_usage_by_layer.png",dpi=200); plt.close(fig)

    # Plot 2: association strengths.
    if not nmis.empty:
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for v,g in nmis.groupby("variable"):
            g=g.sort_values("layer_index")
            ax.plot(g.layer_index,g.nmi_primary_expert,marker="o",label=v)
        ax.set(xlabel="MoE layer index",ylabel="Normalized mutual information",
               title="Routing association with particle/jet observables")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"routing_association_by_layer.png",dpi=200); plt.close(fig)

    # Plot 3+: early/mid/late lift heatmaps for particle type and jet family.
    if not lift.empty:
        for var in ("particle_type","pt_rank_bin","jet_family"):
            sub=lift[lift.variable==var]
            if sub.empty: continue
            ids=sorted(sub.layer_index.unique()); chosen=sorted(set([ids[0],ids[len(ids)//2],ids[-1]]))
            for li in chosen:
                q=sub[sub.layer_index==li].pivot(index="group",columns="expert",values="lift")
                fig,ax=plt.subplots(figsize=(7.0,max(3.2,.42*len(q))))
                im=ax.imshow(q.values,aspect="auto")
                ax.set_xticks(np.arange(len(q.columns))); ax.set_xticklabels([f"E{x}" for x in q.columns])
                ax.set_yticks(np.arange(len(q.index))); ax.set_yticklabels(q.index.astype(str))
                ax.set(xlabel="Expert",ylabel=var,title=f"{var} routing lift — layer {li}")
                fig.colorbar(im,ax=ax,label="P(expert | group) / P(expert)")
                fig.tight_layout(); fig.savefig(out/f"{var}_lift_layer_{li}.png",dpi=200); plt.close(fig)

    # K>1 composition.
    if int(tok.top_k.max())>1 and "expert_1" in tok:
        reg=tok[~tok.is_class_token]
        li=int(reg.layer_index.max()); g=reg[reg.layer_index==li]; ne=int(g.num_experts.max())
        mat=np.zeros((ne,ne),int)
        for a,b in zip(g.expert_0.astype(int),g.expert_1.astype(int)): mat[a,b]+=1
        fig,ax=plt.subplots(figsize=(5.2,4.6)); im=ax.imshow(mat,aspect="auto")
        ax.set(xlabel="Secondary selected expert",ylabel="Primary selected expert",
               title=f"Top-2 expert composition — layer {li}")
        ax.set_xticks(np.arange(ne)); ax.set_yticks(np.arange(ne))
        fig.colorbar(im,ax=ax,label="sampled tokens"); fig.tight_layout()
        fig.savefig(out/"top2_expert_composition_last_layer.png",dpi=200); plt.close(fig)

    highlights={"truth_label_column":label_name,
        "warning":"Descriptive associations only; inspect sample sizes and cross-model/layer stability before calling them specialization."}
    if not nmis.empty:
        highlights["top_routing_associations"]=nmis.sort_values("nmi_primary_expert",ascending=False).head(15).to_dict("records")
    if not jsdf.empty:
        highlights["largest_feature_separations"]=jsdf.sort_values("max_pairwise_js",ascending=False).head(10).to_dict("records")
    if not lift.empty:
        z=lift[np.isfinite(lift.lift)&(lift.n_assignments>=100)]
        highlights["largest_conditional_lifts"]=z.sort_values("lift",ascending=False).head(20).to_dict("records")
    with open(out/"interpretability_highlights.json","w") as f: json.dump(highlights,f,indent=2,default=str)

    print("Analysis complete:",out)

if __name__=="__main__":
    main()
