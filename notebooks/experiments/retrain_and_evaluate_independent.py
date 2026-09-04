"""Retrain corrected tree models and evaluate every model on independent images.

The processed training arrays already exclude the isolated images. Class 1 is PR.
Weighting is applied exactly once: scale_pos_weight for XGBoost and class_weights
for CatBoost. Existing models are preserved under model/; corrected tree models
are written to model/retrained_corrected/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.file_paths import file_paths

SEED = 42
PROCESSED = Path("/Volumes/ep_ssd/database/partialPr/data/processed")
INTERIM = Path("/Volumes/ep_ssd/database/partialPr/data/interim")
MODEL_OUT = ROOT / "model" / "retrained_corrected"
RESULT_OUT = ROOT / "results" / "independent_evaluation_corrected"
MODEL_OUT.mkdir(parents=True, exist_ok=True)
RESULT_OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "3x3": np.array([0, 1, 2, 4, 5, 6, 8, 9, 10]),
    "3x4": np.arange(12),
    "4x3": np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]),
}
ISOLATED = {
    "brain": ["2022-02-16_T_HORAO-1-C_FR_15Z_3", "2022-03-16_T_HORAO-4-D_FR_1_2"],
    "afmmm": ["AFMMM_sample_he9_Data", "AFMMM_sample_bg11_Data", "AFMMM_sample_bg5_Data"],
    "cervix": ["Sample25_550_Data", "Sample2_550_Data", "Sample23_550_Data", "Sample19_550_Data"],
}


class PixelMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        layers=[]; prev=n
        for h in (128, 64, 32):
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(.30)]; prev=h
        layers += [nn.Linear(prev,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def sampled_data():
    """Reproduce the stratified 10% pixel sample without merging 27M feature rows."""
    xs=[]; ys=[]
    for tissue in ("brain", "cervix", "afmmm"):
        x=np.load(PROCESSED/tissue/"merged_all_X.npy", mmap_mode="r").reshape(-1,16)
        y=np.load(PROCESSED/tissue/"merged_all_y.npy", mmap_mode="r").reshape(-1)
        xs.append(x); ys.append(y)
    yall=np.concatenate([np.asarray(y) for y in ys])
    rng=np.random.RandomState(SEED)
    chosen=[]
    for cls in (0,1):
        idx=np.flatnonzero(yall==cls)
        chosen.append(rng.choice(idx, size=len(idx)//10, replace=False))
    idx=np.concatenate(chosen); rng.shuffle(idx)
    offsets=np.cumsum([0]+[len(x) for x in xs])
    X=np.empty((len(idx),16),dtype=np.float32)
    for j,x in enumerate(xs):
        take=(idx>=offsets[j])&(idx<offsets[j+1])
        X[take]=x[idx[take]-offsets[j]]
    return X,yall[idx].astype(np.int8)


def train_trees(X16,y):
    tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
    tr,va=train_test_split(tr,test_size=.25,random_state=SEED,stratify=y[tr])
    weight=float((y[tr]==0).sum()/(y[tr]==1).sum())
    print(f"class 1=PR; weight={weight:.6f}; train/val/test={len(tr)}/{len(va)}/{len(te)}")
    pd.DataFrame([{"positive_class":1,"positive_label":"PR","positive_weight":weight,
                   "train_pixels":len(tr),"validation_pixels":len(va),"internal_test_pixels":len(te)}]).to_csv(
                       RESULT_OUT/"split_and_weight.csv",index=False)
    for scenario,cols in SCENARIOS.items():
        X=X16[:,cols]
        dtr=xgb.DMatrix(X[tr],label=y[tr]); dva=xgb.DMatrix(X[va],label=y[va])
        model=xgb.train(dict(objective="binary:logistic",eval_metric="logloss",eta=.05,max_depth=7,
                             subsample=.8,colsample_bytree=.8,scale_pos_weight=weight,nthread=-1,seed=SEED),
                        dtr,num_boost_round=2000,evals=[(dva,"validation")],early_stopping_rounds=50,
                        verbose_eval=100)
        model.save_model(MODEL_OUT/f"pixel_xgb_{scenario}.json")
        cat=CatBoostClassifier(loss_function="Logloss",eval_metric="Logloss",learning_rate=.05,depth=8,
            iterations=2000,l2_leaf_reg=3,random_seed=SEED,class_weights=[1.0,weight],
            early_stopping_rounds=100,verbose=100)
        cat.fit(X[tr],y[tr],eval_set=(X[va],y[va]))
        cat.save_model(MODEL_OUT/f"pixel_catboost_{scenario}.cbm")
    return weight


def load_mlp(scenario):
    path=file_paths.model_save_path/("best_pixel_mlp.pth" if scenario=="3x4" else f"experiments/{scenario}/best_pixel_mlp.pth")
    model=PixelMLP(len(SCENARIOS[scenario])); model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)); model.eval()
    return model


def predict_mlp(model,X):
    out=[]
    with torch.no_grad():
        for i in range(0,len(X),32768): out.append(torch.sigmoid(model(torch.tensor(X[i:i+32768],dtype=torch.float32))).numpy().ravel())
    return (np.concatenate(out)>.5).astype(np.int8)


def evaluate():
    rows=[]
    for scenario,cols in SCENARIOS.items():
        xb=xgb.Booster(); xb.load_model(MODEL_OUT/f"pixel_xgb_{scenario}.json")
        cb=CatBoostClassifier(); cb.load_model(MODEL_OUT/f"pixel_catboost_{scenario}.cbm")
        mlp=load_mlp(scenario)
        for tissue,samples in ISOLATED.items():
            for sid in samples:
                arr=np.load(INTERIM/tissue/f"{sid}_combined.npy")
                X=arr[...,:16].reshape(-1,16)[:,cols].astype(np.float32); y=arr[...,16].ravel().astype(np.int8)
                predictions={"XGBoost":(xb.predict(xgb.DMatrix(X))>.5).astype(np.int8),
                             "CatBoost":(cb.predict_proba(X)[:,1]>.5).astype(np.int8),
                             "MLP":predict_mlp(mlp,X)}
                for name,p in predictions.items():
                    rows.append(dict(scenario=scenario,model=name,tissue=tissue,sample=sid,n_pixels=len(y),
                        accuracy=accuracy_score(y,p),precision=precision_score(y,p,zero_division=0),
                        recall=recall_score(y,p,zero_division=0),f1=f1_score(y,p,zero_division=0)))
                print("evaluated",scenario,tissue,sid)
    df=pd.DataFrame(rows); df.to_csv(RESULT_OUT/"per_image_metrics.csv",index=False)
    summary=df.groupby(["scenario","model","tissue"])[["accuracy","precision","recall","f1"]].agg(["mean","std","count"])
    summary.to_csv(RESULT_OUT/"mean_sd_by_tissue.csv")
    overall=df.groupby(["scenario","model"])[["accuracy","precision","recall","f1"]].agg(["mean","std","count"])
    overall.to_csv(RESULT_OUT/"mean_sd_overall.csv")
    print(overall.round(4))


if __name__ == "__main__":
    X,y=sampled_data(); train_trees(X,y); evaluate()
