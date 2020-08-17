import numpy as np
import pandas as pd
from  pathlib import  Path
from pandas._libs.lib import is_integer

import weightedstats as ws 
import wquantiles as wq

def wmedian(df, column_name, weights_name='wt0'):
    
    df = df.dropna(subset=[column_name,weights_name ])
    
    return ws.weighted_median( df[column_name], weights=df[weights_name])
    
def wmedian2(df, column_name, weights_name='wt0'):
    
    df = df.dropna(subset=[column_name,weights_name ])
    
    return wq.median( df[column_name], df[weights_name])
    
def wmean(df, column_name, weights_name='wt0'):
    """Calculate the weighted mean of a list."""

    w = df[weights_name]/df[weights_name].sum()
     
    return (df[column_name]*w).sum()



# From https://stackoverflow.com/a/51938636
def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'

    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
        kwargs['labels'] = quantiles[:-1]
    else:
        quantiles = q
    
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    
    if 'retbins' in kwargs:
        return bins[0].sort_index(), bins[1]
    else:       
        return bins.sort_index()

def minimal_corr(df):
    

    tcr = df.corr('spearman')                        # Create a correlation matrix

    # Create an upper triangle matrix, so we can select only part
    # of the correlation matrix
    keep = (np.triu(np.ones(tcr.shape))   # Make a triangular matrix of 1 and 0
            .astype('bool')               # Convert it to bool
            .reshape(tcr.size))           # Turn it from a matrix to a 1D series

    tc = (tcr.stack()                     # Convert from a matris to a 1D series, with rows and col labels as row indexes
            [keep]                        # Select just the rows selected by the trianular matrix
            .to_frame('result'))          # Convert the series into a dataframe

    t = tc[tc.result<1].sort_values('result',ascending=False).reset_index()

    return t[np.abs(t.result) > .1]


def impl_pivot(df, indices, sum_var, wt_var='wt0', agg=wmedian):
    """Aggregate across implicates, perform a weighted mean or median, and 
    return the mean of the per-implicate results. """
    
    if not isinstance(indices,(list, tuple)):
        indices = [indices]
        level = None
    else:
        level = 1
    
    g =  df.groupby(['implicate_id']+indices)
    
    r =  g.apply(agg, sum_var, wt_var).to_frame(sum_var).unstack().mean(level=level)
    
    if level == 1:
        return r
    else:
        return r.to_frame(sum_var)
    
# Create a weighted sample
def make_sample(df, N=10_000_000, extra_cols = []):
    t = df[['case_id', 'record_id', 'race', 'age_1','agecl','norminc','networth','asset', 'gi_sum', 'occat1', 'housecl', 'edcl', 'indcat', 'famstruct', 'married',
            'any_inherit', 'any_transfer'] + extra_cols]
    dfs = t.sample(N,replace=True, weights=df.wt0)

    dfs['income_decile'] = pd.qcut(dfs.norminc, 10, labels=False)  

    # The majority of gi_sum values are 0, so quantiles don't work well. So, compute the
    # bins on only the non-zero values, then apply that to the full series

    o, gi_sum_bins = pd.qcut(dfs[dfs.gi_sum > 0].gi_sum, 10 , retbins = True)
    gi_sum_bins[0] = 0 # So zero gets included in a bin
    dfs['gi_sum_decile']  = pd.cut(dfs.gi_sum, gi_sum_bins, labels=False).fillna(0)
    
    return dfs
