# Master - targeting V0.2

## Breaking

- Changed callbacks to receive `kargs`, rather than logs to allow for great flexibility.

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

## Removals

## Fixes

- Added missing data download cell for multiclass example
- Corrected type hint for `OneCycle lr_range` to `List`
- Corrected `on_train_end` not being called in `fold_train_ensemble`
- Fixed crash in `plot_feat` when plotting non-bulk without cuts, and non-crash bug when plotting non-bulk with cuts
- Fixed typing of callback_args in `fold_train_ensemble`
- Fixed crash when trying to load model trained on cuda device for application on CPU device

## Changes

- Moved `on_train_end` call in `fold_train_ensemble` to after loading best set of weights

## Depreciations

- Callbacks:
    - Added `callback_partials` parameter (a list of partials that yield a Callback object) in `fold_train_ensemble` to eventually replace `callback_args`; Neater appearance than previous Dict of object and kargs
    - `callback_args` now depreciated, to be removed in v0.3
    - Currently `callback_args` are converted to `callback_partials, code will also be removed in v0.3

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
