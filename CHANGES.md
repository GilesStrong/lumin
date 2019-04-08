# Master - targeting V0.1.1

## Breaking

## Potentially breaking

- `binary_class_cut` now returns tuple of `(cut, mean_AMS, maximum_AMS)` as opposed to just the cut
- Initialisation lookups now expected to return callable, rather than callable and dictionary of arguments. `partial` used instead.

## Additions

- Added PReLU activation
- Added uniform initialisation lookup

## Fixes

- `uncert_round` converts `NaN` uncertainty to `0`
- Corect name of internal embedding dropout layer in `CatEmbHead`: emd_do -> emb_do
- Adding missing settings for activations and initialisations to body and tail

## Comments

# V0.1.0 PyPI am assuming direct control

Record of changes begins