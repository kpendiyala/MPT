#!/usr/bin/env python3
import atexit, json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

FEATURE_NAMES = [
    "part_pt_scale_log_norm","part_e_scale_log_norm","part_logptrel_norm",
    "part_logerel_norm","part_deltaR_norm","part_charge","part_isChargedHadron",
    "part_isNeutralHadron","part_isPhoton","part_isElectron","part_isMuon",
    "part_d0","part_d0err","part_dz","part_dzerr","part_deta","part_dphi",
]

def _inv_manual(x, center, scale):
    return x / scale + center

def _hash_keep(event_idx, particle_idx, layer_idx, mod):
    if mod <= 1:
        return np.ones_like(event_idx, dtype=bool)
    e=event_idx.astype(np.uint64); p=particle_idx.astype(np.uint64); l=np.uint64(layer_idx+1)
    h=(e*np.uint64(11400714819323198485)+p*np.uint64(14029467366897019727)+l*np.uint64(1609587929392839161))
    h ^= (h >> np.uint64(29))
    return (h % np.uint64(mod)) == 0

class RoutingInterpretabilityCollector:
    def __init__(self, model, output_dir, run_name=None):
        self.model=model
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name=run_name
        self.sample_mod=int(os.environ.get("INTERP_SAMPLE_MOD","256"))
        self.class_sample_mod=int(os.environ.get("INTERP_CLASS_SAMPLE_MOD","8"))
        self.layers={}; self.layer_order=[]; self.current_valid_masks={}
        self.current_context=None; self.event_offset=0; self.handles=[]
        self.token_chunks=[]; self.class_chunks=[]; self.dumped=False
        self.core_name,self.core=self._find_core()
        self._register_hooks()
        atexit.register(self.dump)

    def _find_core(self):
        c=[]
        for name,m in self.model.named_modules():
            if hasattr(m,"blocks") and hasattr(m,"cls_blocks") and hasattr(m,"moe_num_experts"):
                c.append((name,m))
        if not c:
            raise RuntimeError("Could not locate MoeParticleTransformer core")
        return sorted(c,key=lambda x:len(x[0]),reverse=True)[0]

    def _register_hooks(self):
        print("="*88)
        print("REGISTERING MOE INTERPRETABILITY COLLECTOR")
        print(f"core={self.core_name or '<root>'}")
        print(f"particle sampling ~= 1/{self.sample_mod}; class-token ~= 1/{self.class_sample_mod}")
        print("="*88)
        self.handles.append(self.core.register_forward_pre_hook(self._core_pre_hook, with_kwargs=True))
        for name,m in self.model.named_modules():
            if not (hasattr(m,"router") and hasattr(m,"experts") and hasattr(m,"moe_num_experts") and hasattr(m,"moe_top_k")):
                continue
            li=len(self.layer_order); self.layer_order.append(name)
            n=int(m.moe_num_experts); k=min(int(m.moe_top_k),n)
            self.layers[name]={
                "layer_index":li,"num_experts":n,"top_k":k,
                "capacity_factor":float(m.moe_capacity_factor),
                "aux_loss_coef":float(getattr(m,"moe_aux_loss_coef",0.0)),
                "router_jitter":float(getattr(m,"moe_router_jitter",0.0)),
                "routing_calls":0,"tokens":0,"valid_tokens":0,
                "requested":[0]*n,"accepted":[0]*n,"dropped":[0]*n,
                "valid_requested":[0]*n,"valid_accepted":[0]*n,"valid_dropped":[0]*n,
                "entropy_sum":0.0,"valid_entropy_sum":0.0,
                "top1_prob_sum":0.0,"valid_top1_prob_sum":0.0,
                "selected_weight_sum":[0.0]*n,"accepted_weight_sum":[0.0]*n,
                "pair_counts":defaultdict(int),
            }
            self.handles.append(m.register_forward_pre_hook(self._make_block_pre_hook(name), with_kwargs=True))
            self.handles.append(m.router.register_forward_hook(self._make_router_hook(name,m)))
            print(f"{li:02d} {name}: E={n} K={k} cap={m.moe_capacity_factor} aux={getattr(m,'moe_aux_loss_coef',None)}")
        if not self.layers:
            raise RuntimeError("No MoE blocks found")

    def _core_pre_hook(self,module,args,kwargs):
        x=args[0] if args else kwargs.get("x")
        mask=args[2] if len(args)>2 else kwargs.get("mask")
        if x is None or x.ndim!=3:
            self.current_context=None; return
        batch,channels,seq=x.shape
        if mask is None:
            mask=torch.ones((batch,1,seq),dtype=torch.bool,device=x.device)
        else:
            mask=mask.bool()
        xcpu=x.detach().float().cpu(); mcpu=mask.detach().cpu().squeeze(1).bool()
        ctx={"batch":int(batch),"seq":int(seq),"event_start":int(self.event_offset),"features":xcpu,"mask":mcpu}
        if channels>=17:
            arr=xcpu.numpy()
            logptrel=_inv_manual(arr[:,2,:],-4.7,0.7)
            logerel=_inv_manual(arr[:,3,:],-4.7,0.7)
            dr=_inv_manual(arr[:,4,:],0.2,4.0)
            ptslog=_inv_manual(arr[:,0,:],1.7,0.7)
            eslog=_inv_manual(arr[:,1,:],2.0,0.7)
            ctx.update({
                "part_logptrel":logptrel,"part_logerel":logerel,"part_deltaR":dr,
                "part_ptrel":np.exp(logptrel),"part_erel":np.exp(logerel),
                "part_pt_scale":np.exp(ptslog),"part_e_scale":np.exp(eslog),
            })
            rank=np.full((batch,seq),-1,dtype=np.int16); valid=mcpu.numpy()
            for b in range(batch):
                ids=np.flatnonzero(valid[b])
                if ids.size:
                    order=ids[np.argsort(-logptrel[b,ids],kind="stable")]
                    rank[b,order]=np.arange(1,len(order)+1,dtype=np.int16)
            ctx["pt_rank"]=rank
        self.current_context=ctx
        self.event_offset += int(batch)

    def _make_block_pre_hook(self,layer_name):
        def hook(module,args,kwargs):
            x=args[0] if len(args)>0 else kwargs.get("x")
            x_cls=kwargs.get("x_cls"); pm=kwargs.get("padding_mask")
            if x_cls is not None:
                self.current_valid_masks[layer_name]=torch.ones(x_cls.shape[1],dtype=torch.bool,device=x_cls.device)
            elif pm is None:
                seq,batch,_=x.shape
                self.current_valid_masks[layer_name]=torch.ones(seq*batch,dtype=torch.bool,device=x.device)
            else:
                self.current_valid_masks[layer_name]=(~pm.bool()).transpose(0,1).reshape(-1)
        return hook

    @staticmethod
    def _exact_dispatch(gates,block):
        nt,ne=gates.shape; k=min(int(block.moe_top_k),ne)
        if k==1:
            idx=gates.argmax(dim=-1,keepdim=True); w=gates.gather(1,idx)
            cap=int(float(block.moe_capacity_factor)*math.ceil(nt/max(1,ne)))
        else:
            vals,idx=gates.topk(k=k,dim=-1)
            w=vals/vals.sum(dim=1,keepdim=True).clamp(min=1e-9)
            cap=int(float(block.moe_capacity_factor)*math.ceil((nt*k)/max(1,ne)))
        acc=torch.zeros_like(idx,dtype=torch.bool)
        for e in range(ne):
            rows,cols=torch.nonzero(idx==e,as_tuple=True)
            if rows.numel()==0: continue
            if rows.numel()>cap:
                keep=w[rows,cols].topk(cap,sorted=False).indices
                rows=rows.index_select(0,keep); cols=cols.index_select(0,keep)
            acc[rows,cols]=True
        return idx,w,acc,cap

    def _make_router_hook(self,layer_name,block):
        def hook(router_module,inputs,output):
            with torch.no_grad():
                logits=output.detach()
                if logits.ndim!=2:
                    raise RuntimeError(f"{layer_name}: unexpected router shape {tuple(logits.shape)}")
                gates=torch.softmax(logits,dim=-1)
                idx,w,acc,cap=self._exact_dispatch(gates,block)
                nt,ne=gates.shape; k=idx.shape[1]
                valid=self.current_valid_masks.get(layer_name)
                if valid is None or valid.numel()!=nt:
                    valid=torch.ones(nt,dtype=torch.bool,device=gates.device)
                s=self.layers[layer_name]; s["routing_calls"]+=1; s["tokens"]+=int(nt); s["valid_tokens"]+=int(valid.sum())
                ent=-(gates*gates.clamp(min=1e-12).log()).sum(dim=1); top1=gates.max(dim=1).values
                s["entropy_sum"]+=float(ent.sum()); s["top1_prob_sum"]+=float(top1.sum())
                if valid.any():
                    s["valid_entropy_sum"]+=float(ent[valid].sum()); s["valid_top1_prob_sum"]+=float(top1[valid].sum())
                for slot in range(k):
                    se=idx[:,slot]; sw=w[:,slot]; sa=acc[:,slot]
                    for e in range(ne):
                        req=se==e; ac=req & sa
                        s["requested"][e]+=int(req.sum()); s["accepted"][e]+=int(ac.sum()); s["dropped"][e]+=int((req & ~sa).sum())
                        s["valid_requested"][e]+=int((req & valid).sum()); s["valid_accepted"][e]+=int((ac & valid).sum()); s["valid_dropped"][e]+=int((req & ~sa & valid).sum())
                        if req.any(): s["selected_weight_sum"][e]+=float(sw[req].sum())
                        if ac.any(): s["accepted_weight_sum"][e]+=float(sw[ac].sum())
                if k>1:
                    icpu=idx.cpu().numpy(); vcpu=valid.cpu().numpy()
                    for r in np.flatnonzero(vcpu):
                        s["pair_counts"][tuple(sorted(int(x) for x in icpu[r]))]+=1
                self._sample_rows(layer_name,gates,idx,w,acc,valid,cap)
        return hook

    def _sample_rows(self,layer_name,gates,idx,w,acc,valid,cap):
        s=self.layers[layer_name]; li=s["layer_index"]; is_cls="cls_blocks" in layer_name
        ctx=self.current_context
        if ctx is None: return
        nt=gates.shape[0]; k=idx.shape[1]; batch=ctx["batch"]; seq=ctx["seq"]
        if is_cls:
            if nt!=batch: return
            event=np.arange(ctx["event_start"],ctx["event_start"]+batch,dtype=np.int64)
            particle=np.full(batch,-1,dtype=np.int16)
            keep=_hash_keep(event,np.zeros_like(event),li,self.class_sample_mod)
        else:
            if nt!=batch*seq: return
            event=np.tile(np.arange(ctx["event_start"],ctx["event_start"]+batch,dtype=np.int64),seq)
            particle=np.repeat(np.arange(seq,dtype=np.int16),batch)
            keep=valid.cpu().numpy() & _hash_keep(event,particle.astype(np.int64),li,self.sample_mod)
        rows=np.flatnonzero(keep)
        if rows.size==0: return
        g=gates.float().cpu().numpy(); ii=idx.cpu().numpy(); ww=w.float().cpu().numpy(); aa=acc.cpu().numpy()
        ent=-(g[rows]*np.log(np.clip(g[rows],1e-12,None))).sum(axis=1)
        data={
            "event_index":event[rows],"layer":np.full(rows.size,layer_name),"layer_index":np.full(rows.size,li,dtype=np.int16),
            "is_class_token":np.full(rows.size,is_cls),"particle_index":particle[rows],
            "num_experts":np.full(rows.size,g.shape[1],dtype=np.int16),"top_k":np.full(rows.size,k,dtype=np.int16),
            "capacity":np.full(rows.size,cap,dtype=np.int32),"router_entropy":ent,
            "router_entropy_norm":ent/max(1e-12,math.log(g.shape[1])),"top1_probability":g[rows].max(axis=1),
        }
        for slot in range(k):
            data[f"expert_{slot}"]=ii[rows,slot].astype(np.int16); data[f"weight_{slot}"]=ww[rows,slot]; data[f"accepted_{slot}"]=aa[rows,slot]
        if not is_cls:
            bi=rows % batch; pi=rows // batch; feat=ctx["features"].numpy()
            for j,name in enumerate(FEATURE_NAMES):
                if j<feat.shape[1]: data[name]=feat[bi,j,pi]
            for name in ("part_logptrel","part_logerel","part_deltaR","part_ptrel","part_erel","part_pt_scale","part_e_scale","pt_rank"):
                if name in ctx: data[name]=ctx[name][bi,pi]
            if feat.shape[1]>=11:
                onehot=feat[bi,6:11,pi]
                names=np.array(["charged_hadron","neutral_hadron","photon","electron","muon"],dtype=object)
                best=np.argmax(onehot,axis=1); vmax=onehot[np.arange(onehot.shape[0]),best]
                ptype=names[best].astype(object); ptype[vmax<0.5]="other"; data["particle_type"]=ptype
        df=pd.DataFrame(data)
        (self.class_chunks if is_cls else self.token_chunks).append(df)

    @staticmethod
    def _div(a,b): return None if b==0 else a/b

    def dump(self):
        if self.dumped: return
        self.dumped=True
        layers=[]; experts=[]; pairs=[]
        for name in self.layer_order:
            s=self.layers[name]; n=s["num_experts"]
            req=sum(s["requested"]); acc=sum(s["accepted"]); drop=sum(s["dropped"])
            vreq=sum(s["valid_requested"]); vacc=sum(s["valid_accepted"]); vdrop=sum(s["valid_dropped"])
            vf=[self._div(x,vreq) or 0.0 for x in s["valid_requested"]]; ideal=1/n
            layers.append({
                "layer":name,"layer_index":s["layer_index"],"num_experts":n,"top_k":s["top_k"],
                "capacity_factor":s["capacity_factor"],"aux_loss_coef":s["aux_loss_coef"],"router_jitter":s["router_jitter"],
                "routing_calls":s["routing_calls"],"tokens":s["tokens"],"valid_tokens":s["valid_tokens"],
                "requested_assignments":req,"accepted_assignments":acc,"dropped_assignments":drop,
                "drop_fraction":self._div(drop,req),"valid_requested_assignments":vreq,"valid_accepted_assignments":vacc,
                "valid_dropped_assignments":vdrop,"valid_drop_fraction":self._div(vdrop,vreq),
                "mean_router_entropy":self._div(s["entropy_sum"],s["tokens"]),
                "mean_valid_router_entropy":self._div(s["valid_entropy_sum"],s["valid_tokens"]),
                "mean_normalized_valid_router_entropy":(self._div(s["valid_entropy_sum"],s["valid_tokens"])/math.log(n) if s["valid_tokens"] and n>1 else None),
                "mean_top1_probability":self._div(s["top1_prob_sum"],s["tokens"]),
                "mean_valid_top1_probability":self._div(s["valid_top1_prob_sum"],s["valid_tokens"]),
                "valid_load_imbalance_ratio":(max(vf)/ideal if vreq else None),
            })
            for e in range(n):
                experts.append({
                    "layer":name,"layer_index":s["layer_index"],"expert":e,
                    "requested":s["requested"][e],"accepted":s["accepted"][e],"dropped":s["dropped"][e],
                    "valid_requested":s["valid_requested"][e],"valid_accepted":s["valid_accepted"][e],"valid_dropped":s["valid_dropped"][e],
                    "valid_requested_fraction":self._div(s["valid_requested"][e],vreq),
                    "valid_accepted_fraction":self._div(s["valid_accepted"][e],vacc),
                    "mean_selected_weight":self._div(s["selected_weight_sum"][e],s["requested"][e]),
                    "mean_accepted_weight":self._div(s["accepted_weight_sum"][e],s["accepted"][e]),
                })
            for combo,count in sorted(s["pair_counts"].items()):
                pairs.append({"layer":name,"layer_index":s["layer_index"],"expert_combination":",".join(map(str,combo)),"count":count})
        pd.DataFrame(layers).to_csv(self.output_dir/"layer_summary.csv",index=False)
        pd.DataFrame(experts).to_csv(self.output_dir/"expert_usage.csv",index=False)
        if pairs: pd.DataFrame(pairs).to_csv(self.output_dir/"topk_combinations.csv",index=False)
        if self.token_chunks:
            pd.concat(self.token_chunks,ignore_index=True).to_parquet(self.output_dir/"token_sample.parquet",index=False,compression="zstd")
        if self.class_chunks:
            pd.concat(self.class_chunks,ignore_index=True).to_parquet(self.output_dir/"class_token_sample.parquet",index=False,compression="zstd")
        manifest={
            "run_name":self.run_name,"core_module":self.core_name,"num_moe_blocks":len(self.layers),
            "events_seen":self.event_offset,"particle_sample_mod":self.sample_mod,"class_sample_mod":self.class_sample_mod,
            "feature_names":FEATURE_NAMES,
            "routing_semantics":{
                "top1_capacity":"capacity_factor * ceil(tokens / E)",
                "topk_capacity":"capacity_factor * ceil(tokens * K / E)",
                "topk_weights":"selected softmax gates renormalized to sum to 1 per token",
                "overflow":"highest selected routing weights retained independently per expert",
                "jitter":"inactive in model.eval()/prediction mode",
            }
        }
        with open(self.output_dir/"interpretability_manifest.json","w") as f: json.dump(manifest,f,indent=2)
        print("="*88); print("INTERPRETABILITY COLLECTION COMPLETE"); print(self.output_dir); print("="*88)
