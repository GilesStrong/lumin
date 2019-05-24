# Matser - Targeting V0.2.1

## Important changes

## Breaking

## Additions

## Removals

## Fixes

## Changes

## Depreciations

## Comments

# V0.2 Bonfire Lit

## Important changes

- Residual mode in `FullyConnected`:
    - Identity paths now skip two layers instead of one to align better with [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
    - In cases where an odd number of layers are specified for the body, the number of layers is increased by one
    - Batch normalisation now corrected to be after the addition step (previously was set before)
- Dense mode in `FullyConnected` now no longer adds an extra layer to scale down to the original width, instead `get_out_size` now returns the width of the final concatinated layer and the tail of the network is expected to accept this input size
- Fixed rule-of-thumb for embedding sizes from max(50, 1+(sz//2)) to max(50, (1+sz)//2)

## Breaking

- Changed callbacks to receive `kargs`, rather than logs to allow for great flexibility
- Residual mode in `FullyConnected`:
    - Identity paths now skip two layers instead of one to align better with [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
    - In cases where an odd number of layers are specified for the body, the number of layers is increased by one
    - Batch normalisation now corrected to be after the addition step (previously was set before)
- Dense mode in `FullyConnected` now no longer adds an extra layer to scale down to the original width, instead `get_out_size` now returns the width of the final concatinated layer and the tail of the network is expected to accept this input size
- Initialisation arguments for `CatEmbHead` changed considerably w.r.t. embedding arguments; now expects to receive a `Embedder` class

## Additions

- Added wrapper class for significance-based losses (`SignificanceLoss`)
- Added label smoothing for binary classification
- Added `on_eval_begin` and `on_eval_end` callback calls
- Added `on_backwards_begin` and `on_backwards_end` callback calls
- Added callbacks to `fold_lr_find`
- Added gradient-clipping callback
- Added default momentum range to `OneCycle` of .85-.95
- Added `SequentialReweight` classes
- Added option to turn of realtime loss plots
- Added `from_results` and `from_save` classmethods for `Ensemble`
- Added option to `SWA` to control whether it only updates on cycle end when paired with an `AbsCyclicalCallback`
- Added helper class `Embedder` to simplify parsing of embedding settings
- Added parameters to save and configure plots to `get_nn_feat_importance`, `get_ensemble_feat_importance`, and `rf_rank_features`
- Added classmethod for `Model` to load from save
- Added experimental export to Tensorflow Protocol Buffer

## Removals

## Fixes

- Added missing data download cell for multiclass example
- Corrected type hint for `OneCycle lr_range` to `List`
- Corrected `on_train_end` not being called in `fold_train_ensemble`
- Fixed crash in `plot_feat` when plotting non-bulk without cuts, and non-crash bug when plotting non-bulk with cuts
- Fixed typing of callback_args in `fold_train_ensemble`
- Fixed crash when trying to load model trained on cuda device for application on CPU device
- Fixed positioning of batch normalisation in residual mode of `FullyConnected` to after addition
- `rf_rank_features` was accidentally evaluating feature importance on validation data rather than training data, resulting in lower importances that it should
- Fixed feature selection in examples using a test size of 0.8 rather than 0.2
- Fixed crash when no importnat features were found by `rf_rank_features`
- Fixed rule-of-thumb for embedding sizes from max(50, 1+(sz//2)) to max(50, (1+sz)//2)
- Fixed cutting when saving plots as pdf

## Changes

- Moved `on_train_end` call in `fold_train_ensemble` to after loading best set of weights
- Replaced all mutable default arguments

## Depreciations

- Callbacks:
    - Added `callback_partials` parameter (a list of partials that yield a Callback object) in `fold_train_ensemble` to eventually replace `callback_args`; Neater appearance than previous Dict of object and kargs
    - `callback_args` now depreciated, to be removed in v0.3
    - Currently `callback_args` are converted to `callback_partials`, code will also be removed in v0.3
- Embeddings:
    - Added `cat_embedder` parameter to `ModelBuilder` to eventuall replace `cat_args`
    - `cat_args` now depreciated to be removed in v0.3
    - Currently `cat_args` are converted to an `Embedder`, code will also be removed in v0.3

## Comments

# V0.1.1 PyPI am assuming direct control - micro update

## Breaking

- `binary_class_cut` now returns tuple of `(cut, mean_AMS, maximum_AMS)` as opposed to just the cut
- Initialisation lookups now expected to return callable, rather than callable and dictionary of arguments. `partial` used instead.
- `top_perc` in `binary_class_cut` now treated as percentage rather than fraction

## Additions

- Added PReLU activation
- Added uniform initialisation lookup

## Removals

## Fixes

- `uncert_round` converts `NaN` uncertainty to `0`
- Correct name of internal embedding dropout layer in `CatEmbHead`: emd_do -> emb_do
- Adding missing settings for activations and initialisations to body and tail
- Corrected plot annotation for percentage in `binary_class_cut`

## Changes

- Removed the `BatchNorm1d` automatically added in `CatEmbHead` when using categorical inputs; assuming unit-Gaussian continuous inputs, no *a priori* resaon to add it, and tests indicated it hurt performance and train-time.
- Changed weighting factor when not loading loading cycles only to n+2 from n+1

## Depreciations

## Comments

# V0.1.0 PyPI am assuming direct control

Record of changes begins
