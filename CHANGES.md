# Master - targeting V0.1.2

## Breaking

## Additions

## Removals

## Fixes

- Added missing data download cell for multiclass example

## Changes

## Depreciations

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