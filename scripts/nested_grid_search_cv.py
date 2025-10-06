
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from loaders import load_nifti_image, create_generator
from pyment.models import RegressionSFCN
from pyment.utils import load_select_pretrained_weights

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

## create model function

  #import tensorflow as tf
  #from sklearn.linear_model import LinearRegression
  #from pyment.models.sfcn import RegressionSFCN
  #from pyment.utils import load_select_pretrained_weights


def build_model(model_name,config, weights=None):
  assert(config.keys() >= {'learning_rate','weight_decay','dropout'}) # confirming the hyperparameters are present
  if model_name == 'RegressionSFCN':
    MODEL = RegressionSFCN()
    model = load_select_pretrained_weights(MODEL, weights, 'age')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        weight_decay = config['weight_decay']
        )

    model.compile(
        optimizer = optimizer,
        loss = 'mse',
        metrics = ['mae']
        # forward_pass=True when debugging
        )
  elif model_name == 'test':
    model = LinearRegression() # LinearRegression from sklearn does not have compile method
    # This part needs to be adjusted based on how you plan to use LinearRegression
    # For now, I'll keep it as is but note that .compile() is not valid for it.
    # model.compile(
    #     optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate']),
    #     loss = 'mse',
    #     metrics = ['mae']
    # )
  return model

# Returns all combinations of hyperparameters from a dictionary of choices
from itertools import product
def create_hyperparameter_configurations(hyperparameters):
  keys = hyperparameters.keys()
  values = hyperparameters.values()
  configurations = [dict(zip(keys,combo)) for combo in product(*values)]
  len_configurations = len(configurations)
  print(len_configurations)
  return configurations

def train_configuration(df, weights, configuration, checkpoint_nr,epochs, mode):
  if mode =='print':
    print('Hyperparameter configuration:')
    for param_name, param_value in configuration.items():
      print(f'{param_name}= {param_value}')

  elif mode =='test':
    print('test started')
    model = build_model('test',configuration)
    model.fit(
        df['score'],
        df['age'],
        epochs = epochs,
        callbacks =[cb_checkpoint, cb_earlystopo, reduce_lr]
    )
    print('test completed')
    return model # Return the model

  elif mode == 'train':
    model = build_model('RegressionSFCN',configuration, weights) # Pass weights
    # Assuming create_generator is defined elsewhere and works with your df
    # You need to define cb_checkpoint, cb_earlystopo, reduce_lr
    # Example placeholders:
    cb_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    cb_earlystopo = EarlyStopping(patience=10, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss')

    model.fit(
        create_generator(df),
        epochs = epochs,
        callbacks =[cb_checkpoint, cb_earlystopo, reduce_lr],
    )
    return model


# Cross validation script:
'''
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from pyment.models import RegressionSFCN
from pyment.utils import load_select_pretrained_weights
'''

def run_cross_validation(df,strat_var,target_var, outer_folds, inner_folds,hyperparameters, mode):
  configurations = create_hyperparameter_configurations(hyperparameters)
  outer_splits = folds_stratify_split(df,outer_folds,strat_var) # Corrected variable name

  outer_results= []

  # outer cross-valildation loop
  for fold in range(len(outer_splits)): # Iterate over indices
    outer_fold_number= fold
    print (f'Outer fold nr:{outer_fold_number}')
    outer_train, outer_test = Cross_val_splits(outer_splits,fold)
    inner_splits = folds_stratify_split(outer_train, inner_folds, strat_var) # Split outer_train
    inner_results=[]

     # inner loop
    for inner_fold_idx in range(len(inner_splits)): # Iterate over indices
      inner_fold_number= [outer_fold_number,inner_fold_idx]
      print (f'Inner fold nr:{inner_fold_number}')
      inner_train, inner_val = Cross_val_splits(inner_splits,inner_fold_idx) # Changed inner_test to inner_val


      best_mae = float('inf')
      best_config = None
      # maes = []
      # df of configurations, runs, performance,
      # hyperparameter tuning loop, similar to sklearn. grid_searchCV
      for configuration in configurations:
        model = train_configuration(inner_train, configuration=configuration, weights=None, checkpoint_nr=None, epochs=50, mode=mode) # weights and checkpoint_nr are None for now
        if mode =='train':
          predictions = model.predict(create_generator(inner_val))
          mae = tf.keras.metrics.mean_absolute_error(inner_val[target_var], predictions).numpy()
        elif mode =='prototype':
          predictions = model.predict(inner_val[target_var])
          mae = tf.keras.metrics.mean_absolute_error(inner_val[target_var], predictions).numpy()
        elif mode == 'print':
          # Placeholder for evaluation - replace with actual model evaluation
          # For demonstration, let's assume a dummy mae calculation
          dummy_predictions = np.random.rand(len(inner_val)) * 100 # Dummy predictions
          mae = np.mean(np.abs(inner_val[target_var] - dummy_predictions)) # Dummy MAE
        inner_results.append({'config': configuration, 'mae': mae})

        # Track best configuration for this inner fold
        if mae < best_mae:
            best_mae = mae
            best_config = configuration

      # After iterating through all configs for the inner fold, best config is stored, maybe rerun on larger part or just log results, ask
      print(f"Best config for inner fold {inner_fold_number}: {best_config} with MAE: {best_mae}")
      # train_final_model(pd.concat([inner_train, inner_val]), best_config)
      # final_predictions = final_model.predict(create_generator(outer_test))
      # outer_test_mae = tf.keras.metrics.mean_absolute_error(outer_test['score'], final_predictions).numpy()
      # outer_results.append({'outer_fold': outer_fold_number, 'best_config': best_config, 'test_mae': outer_test_mae})