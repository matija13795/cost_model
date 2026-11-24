import ast
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
ROOT=Path(__file__).resolve().parents[2]
from models.gnn.evaluation.paper_metrics import calculate


def upstream_function(name, file, namespace):
    tree=ast.parse((ROOT/file).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),namespace)
    return namespace[name]


def test_weighting_is_by_schedule():
    frame=pd.DataFrame(dict(sample_id=['a0','b0','b1','b2'],name=['a','b','b','b'],target=[1.,1.,1.,1.],prediction=[2.,1.,1.,1.]))
    result,_=calculate(frame)
    assert result['MAPE_instance_percent']==25
    assert result['MAPE_program_percent']==50
    assert result['undefined_nDCG']==1 and result['undefined_Spearman']==2


def test_against_pinned_matija_functions_with_ties_and_singletons():
    frame=pd.DataFrame(dict(sample_id=[str(i) for i in range(8)], name=['a']*4+['b']*3+['c'],
                            target=[1.,2.,3.,4.,1.,1.,1.,2.],prediction=[1.,1.,2.,3.,2.,2.,2.,3.]))
    result,programs=calculate(frame)
    criterion=upstream_function('mape_criterion','utils/train_utils.py',{'torch':torch})
    expected=criterion(torch.tensor(frame.prediction.to_numpy(),dtype=torch.float32),torch.tensor(frame.target.to_numpy(),dtype=torch.float32)).item()
    assert abs(result['MAPE_instance_percent']-expected)<1e-5
    namespace={'pd':pd,'np':np,'spearmanr':spearmanr,'ndcg_score':ndcg_score}
    for upstream,ours in [('function_wise_ndcg_full','nDCG'),('function_wise_ndcg_5','nDCG_5'),('function_wise_ndcg_1','nDCG_1'),('function_wise_spearman','Spearman')]:
        function=upstream_function(upstream,'utils/data_utils.py',namespace)
        for name,g in frame.groupby('name'):
            g=g.copy()
            for col in ['target','prediction']: g[col]=np.around(g[col].to_numpy(dtype=np.float32),6)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); expected=function(g).iloc[0]
            actual=programs.loc[programs.name==name,ours].iloc[0]
            assert (np.isnan(expected) and np.isnan(actual)) or expected==actual


def test_full_ndcg_not_top_ten_and_input_rejection():
    frame=pd.DataFrame(dict(sample_id=[str(i) for i in range(20)],name=['p']*20,target=np.arange(1,21),prediction=np.arange(20,0,-1)))
    result,_=calculate(frame)
    assert result['nDCG_full']!=ndcg_score([frame.target],[frame.prediction],k=10)
    assert result['Spearman']==-1
    frame.loc[0,'sample_id']='1'
    with pytest.raises(ValueError,match='unique'): calculate(frame)
    frame.loc[0,'sample_id']='0';frame.loc[0,'target']=0
    with pytest.raises(ValueError,match='positive'): calculate(frame)


def test_top_k_mean_and_singleton_counts():
    frame=pd.DataFrame(dict(sample_id=['a','b','c','d'],name=['p','p','p','q'],target=[1.,2.,4.,1.],prediction=[1.,3.,2.,1.]))
    result,_=calculate(frame)
    assert result['nDCG_1']==0.5
    assert result['nDCG_5']==result['nDCG_full']
    assert result['undefined_nDCG_1']==1 and result['undefined_nDCG_5']==1
