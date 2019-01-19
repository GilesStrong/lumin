import numpy as np
from typing import Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastprogress import progress_bar
from sklearn.metrics import roc_auc_score

from ..nn.data.fold_yielder import FoldYielder
from ..optimisation.features import get_rf_feat_importance
from .statistics import uncert_round


def check_val_set(train:FoldYielder, val:FoldYielder, test:Optional[FoldYielder]=None, n_folds:Optional[int]=None) -> None:
    n = min(train.n_folds, val.n_folds)
    if test is not None: n = min(n, test.n_folds)
    if n_folds is None:  n = min(n, n_folds)
        
    samples = {'train': train} if test is None else {'train': train, 'test': test}
    for sample in samples:
        aucs = []
        fi = pd.DataFrame()
        for fold_id in progress_bar(range(n_folds)):
            df_0 = samples[sample].get_df(inc_inputs=True, deprocess=True, fold_id=fold_id, verbose=False)
            df_1 = val.get_df(inc_inputs=True, deprocess=True, fold_id=fold_id, verbose=False)
            df_0['gen_target'] = 0
            df_1['gen_target'] = 1
            df_0['gen_weight'] = 1/len(df_0)
            df_1['gen_weight'] = 1/len(df_1)

            df = df_0.append(df_1, ignore_index=True).sample(frac=1)
            df_trn, df_val = df[:len(df)//2], df[len(df)//2:]

            m = RandomForestClassifier(n_estimators=40, min_samples_leaf=25, n_jobs=-1)
            m.fit(df_trn, df_trn['gen_target'], df_trn['gen_weight'])
            aucs.append(roc_auc_score(df_val['gen_target'], m.predict(df_val), sample_weight=df_val['gen_weight']))
            fi = fi.append(get_rf_feat_importance(m, df_val, df_val['gen_target'], df_val['gen_weight']), ignore_index=True)

        mean = uncert_round(np.mean(aucs), np.std(aucs, ddof=1)/np.sqrt(len(aucs)))
        print(f"\nAUC for {sample}-validation discrimination = {mean[0]}±{mean[1]}")

        print("Top 10 most important features are:")
        mean_fi = pd.DataFrame()
        mean_fi['Importance'] = fi['Importance'].groupby(fi['Feature']).mean()
        mean_fi['Uncertainty'] = fi['Importance'].groupby(fi['Feature']).std()/np.sqrt(n)
        mean_fi.sort_values(['Importance'], inplace=True, ascending=False)
        mean_fi.reset_index(inplace=True)
        print(mean_fi[:min(10, len(mean_fi))])