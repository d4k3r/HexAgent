"""Corrected Group49-style PyTorch student trainer primitives."""
from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import Dataset

from .schema import TrainingExample, read_jsonl
from .validation import validate_example

BOARD=11; AREA=121; DATASET_ID="group49-reconstructed-katahex-teacher-pilot-v2-diversified-gpu"

class ResidualBlock(nn.Module):
    def __init__(self,c:int):
        super().__init__(); self.c1=nn.Conv2d(c,c,3,padding=1); self.b1=nn.BatchNorm2d(c); self.c2=nn.Conv2d(c,c,3,padding=1); self.b2=nn.BatchNorm2d(c); self.r=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.r(self.b2(self.c2(self.r(self.b1(self.c1(x)))))+x)

class Group49Student(nn.Module):
    """Exact final historical module topology; 10x256 is a reconstructed config."""
    def __init__(self, channels:int=256, blocks:int=10):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(6,channels,3,padding=1),nn.BatchNorm2d(channels),nn.ReLU(inplace=True)); self.blocks=nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy=nn.Sequential(nn.Conv2d(channels,2,1),nn.BatchNorm2d(2),nn.ReLU(inplace=True),nn.Flatten(),nn.Linear(2*AREA,AREA))
        self.value=nn.Sequential(nn.Conv2d(channels,1,1),nn.BatchNorm2d(1),nn.ReLU(inplace=True),nn.Flatten(),nn.Linear(AREA,64),nn.ReLU(inplace=True),nn.Linear(64,1),nn.Tanh())
    def forward(self,x):
        x=self.blocks(self.stem(x)); return self.policy(x),self.value(x).squeeze(-1)

def soft_policy_loss(logits, pi, weights):
    """Weighted -sum(pi * log_softmax), never class-index/argmax CE."""
    per=-(pi*torch.log_softmax(logits,dim=1)).sum(dim=1)
    denom=weights.sum().clamp_min(1.0); return (per*weights).sum()/denom

def weighted_mse(pred,z,weights):
    return (((pred-z).square())*weights).sum()/weights.sum().clamp_min(1.0)

def _digest(value:object)->str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def balanced_game_split(games:Iterable[tuple[str,bool]],dataset_id:str=DATASET_ID, train_per_swap:int=24)->dict:
    grouped={False:[],True:[]}
    for gid,swap in games: grouped[bool(swap)].append(gid)
    result={}
    for swap,ids in grouped.items():
        ordered=sorted(ids,key=lambda gid:hashlib.sha256(f"{dataset_id}:{gid}".encode()).hexdigest())
        if not 0 < train_per_swap < len(ordered): raise ValueError("train_per_swap must leave validation games")
        for gid in ordered[:train_per_swap]: result[gid]="train"
        for gid in ordered[train_per_swap:]: result[gid]="validation"
    return result

def atomic_json(path:Path,value:object):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)

def audit_and_load(root:Path, split:dict[str,str], mode:str="phase-a-teacher") -> tuple[list[TrainingExample],dict]:
    examples=[]; counts={"rows":0,"validated":0,"phase_a_teacher":0,"completion":0,"terminal":0,"policy_rows":0,"value_rows":0,"fields":set()}
    for gid,part in split.items():
        path=root/'games'/gid/'examples.jsonl'
        for ex in read_jsonl(path):
            validate_example(ex); counts['rows']+=1;counts['validated']+=1;counts['fields'].update(ex.to_dict())
            if ex.source=='katahex_teacher' and ex.position_status=='normal': counts['phase_a_teacher']+=1
            if ex.source=='completion': counts['completion']+=1
            if ex.position_status=='literal_terminal':counts['terminal']+=1
            use=ex.source=='katahex_teacher' and ex.position_status=='normal' if mode=='phase-a-teacher' else ex.position_status!='literal_terminal'
            if use:
                examples.append(ex); counts['policy_rows']+= int(ex.policy.pi is not None and ex.policy.weight>0); counts['value_rows']+=int(ex.value.z is not None and ex.value.weight>0)
    counts['fields']=sorted(counts['fields']); counts['selected_rows']=len(examples); return examples,counts

def audit_and_load_sources(sources:dict[str,Path], split:dict[str,str], mode:str="phase-a-teacher") -> tuple[list[TrainingExample],dict]:
    """Load split-assigned game files from immutable corpus roots."""
    examples=[]; counts={"rows":0,"validated":0,"phase_a_teacher":0,"completion":0,"terminal":0,"policy_rows":0,"value_rows":0,"fields":set()}
    for gid in sorted(split):
        if gid not in sources: raise ValueError(f"missing corpus source for {gid}")
        path=sources[gid]/'games'/gid/'examples.jsonl'
        for ex in read_jsonl(path):
            validate_example(ex); counts['rows']+=1;counts['validated']+=1;counts['fields'].update(ex.to_dict())
            if ex.source=='katahex_teacher' and ex.position_status=='normal': counts['phase_a_teacher']+=1
            if ex.source=='completion': counts['completion']+=1
            if ex.position_status=='literal_terminal': counts['terminal']+=1
            use=ex.source=='katahex_teacher' and ex.position_status=='normal' if mode=='phase-a-teacher' else ex.position_status!='literal_terminal'
            if use:
                examples.append(ex); counts['policy_rows']+=int(ex.policy.pi is not None and ex.policy.weight>0); counts['value_rows']+=int(ex.value.z is not None and ex.value.weight>0)
    counts['fields']=sorted(counts['fields']);counts['selected_rows']=len(examples);return examples,counts

class ExampleDataset(Dataset):
    """Examples optionally paired with their exact colour-transpose symmetry.

    The transformed half is constructed through ``transform_example``: its
    connection planes come from a freshly rebuilt authoritative HexBoard/DSU.
    """
    def __init__(self,examples:list[TrainingExample], *, symmetry_augment:bool=False):
        self.examples=examples; self.symmetry_augment=bool(symmetry_augment)
    def __len__(self): return len(self.examples)*(2 if self.symmetry_augment else 1)
    def __getitem__(self,i):
        if not 0 <= i < len(self): raise IndexError(i)
        transformed=self.symmetry_augment and i>=len(self.examples)
        x=self.examples[i-len(self.examples)] if transformed else self.examples[i]
        if transformed:
            from .symmetry import transformed_training_tensors
            state,pi,legal,z=transformed_training_tensors(x)
        else:
            state=x.state.planes; pi=x.policy.pi or [0.0]*AREA; legal=x.policy.legal_mask; z=x.value.z
        return {"state":torch.tensor(state,dtype=torch.float32).reshape(6,BOARD,BOARD),"pi":torch.tensor(pi,dtype=torch.float32),"legal_mask":torch.tensor(legal,dtype=torch.bool),"policy_weight":torch.tensor(x.policy.weight,dtype=torch.float32),"z":torch.tensor(z,dtype=torch.float32),"value_weight":torch.tensor(x.value.weight,dtype=torch.float32),"game_id":x.game_id,"symmetry_transformed":torch.tensor(transformed)}

def architecture_manifest(model:nn.Module, *,channels:int,blocks:int)->dict:
    count=sum(p.numel() for p in model.parameters()); return {"architecture":"group49-final-residual-policy-value","input_shape":[6,11,11],"policy_logits":121,"value":"tanh scalar","channels":channels,"residual_blocks":blocks,"parameters":count,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}

def deterministic_milestone_checkpoint_spec() -> dict:
    """Fixed reporting milestone; it does not alter selection or optimization."""
    return {"epoch":12,"filename":"epoch-12.pt","written_when_run_reaches_epoch":12}
