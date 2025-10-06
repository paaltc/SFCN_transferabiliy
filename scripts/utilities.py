import pandas as pd
import numpy as np
def folds_stratify_split(df,folds,strat_var):
  sorted = df.sort_values(by=strat_var)
  df_fold= np.arange(len(df)) % folds
  splits =  [] # list of dfs, one for each fold
  for fold in range(folds):
    fold_subset = df[df_fold == fold].reset_index(drop=True)
    print(fold_subset.head())
    splits.append(fold_subset)
    print(len(fold_subset))
  return splits # list containing split dfs

def Cross_val_splits(splits,fold):
  df_test = splits[fold]
  df_train = []
  for i, x in enumerate(splits):
    if i != fold:
      df_train.append(x)
  df_train = pd.concat(df_train).reset_index(drop=True)
  return df_train,df_test
