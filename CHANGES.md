# Targeting V0.3.1

## Important changes

- Online documentation now available at https://lumin.readthedocs.io

## Breaking

## Additions

- `bin_binary_class_pred`
    - Ability to only consider classes rather than samples when computing bin edges
    - Ability to add pure signal bins if normalisation uncertainty would be below some value
- `plot_bottleneck_weighted_inputs` method for interpretting bottleneck blocks in `MultiBlock`
- Online documentation: https://lumin.readthedocs.io
- Default optimiser notice
- Can now pass arbitary optimisers to the 'opt' value in `opt_args`. Optimisers still interpretable from strings.

## Removals

- weak decorators for losses

## Fixes

- `CatEmbedder.from_fy` using features ignored by `FoldYielder`
- `bottleneck_sz_masks` to `bottleneck_masks` in `MultiBlock`
- SWA crahsing when evaluating targets of type long, when loss expects a float (model.evaluate now converts to float when objective is not multiclass classification)
- Doc string fixes
- Fixed model being moved to device after instantiating optimiser (sometimes leads to an error). Models now moved to device in `ModelBuilder.get_model` rather than in `Model.__init__`

## Changes

## Depreciations

## Comments

# V0.3 Tears in Rain

## Important changes

- `norm_in` default value for `get_pre_proc_pipes` is now `True` rather than `False`
- layer width in dense=True `FullyConnected` now no longer scales with input size to prevent parameter count from exploding
- Biases in `FullyConnected` linear layers are now initialised to zero, rather that default PyTorch init
- Bias in `ClassRegMulti` linear layer is now intitialised to 0.5 if sigmoid output, zero if linear output, and 1/n_out if softmax, unless a bias_init value is specified

## Breaking

- Changed order of arugments in `AMS` and `MultiAMS` and removed some default values
- Removed default for `return_mean` in `RegAsProxyPull` and `RegPull`
- Changed`settings` to `plot_settings` in `rf_rank_features`
- Removed some default parameters for NN blocks in `ModelBuilder`
- `ModelBuilder` `model_args` should now be a dictionary of dictionaries of keyword arguments, one for head, body, and tail blocks,
    previously was a single dictionary of keyword arguments
- `Embedder.from_fy` now no longer works: change to `CatEmbedder.from_fy`
- `CatEmbHead` now no longer has a `n_cont_in` argument, instead one should pass a list of feature names to `cont_feats`

## Additions

- Added `n_estimators` parameter to `rf_rank_features` and `get_opt_rf_params` to adjust the number of trees
- Added `n_rfs` parameter to `rf_rank_features` to average feature importance over several random forests
- Added automatic computation of 3-momenta magnitude to `add_mass` if it's missing
- `n_components` to `get_pre_proc_pipes` to be passed to `PCA`
- `Pipeline` configuration parameters to `fit_input_pipe`
- Ability to pass an instantiated `Pipeline` to `fit_input_pipe`
- Callbacks now receive `model_num` and `savepath` in `on_train_begin`
- Random Forest *like* ensembling:
    - `BootstrapResample` callback for resampling training and validation data
    - Feature subsambling:
        - `FeatureSubsample` callback for training on random selection of features
        - `Model` now has an `input_mask` to automatically mask inputs at inference time (train-time inputs should be masked at `BatchYielder` level)
- `plot_roc` now returns aucs as dictionary
- growth_rate scaling coefficient to `FullyConnected` to adjust layer width by depth
- `n_in` parameter to `FullyConnected` so it works on arbitray size inputs
- `freeze_tail` to `ModelBuilder` and `ClassRegMulti`
- Abstract blocks for head, body, and tail
- `cont_feats` argument to `ModelBuilder` to allow passing of list of named features, eventually allowing more advanced methods based on named outputs of head blocks.
- `CatEmbHead` now computes a mapping from named input features to their outputs
- body blocks now expect to be passed a dictionary mapping from named input features to the model to the outputs of the head block
- `Model` and `AbsBlock` classes now have a method to compute total number of (trainable) parameters
- `MultiBlock` body, providing possibility for multiple, parallel body blocks taking subsets of input features
- Explicit initialisation paramater for bias in `ClassRegMulti`
- `plot_1d_partial_dependence` now takes `pdp_isolate_kargs` and `pdp_plot_kargs` to pass to `pdp_isolate` and  `pdp_plot`, respectively
- `plot_2d_partial_dependence` now takes `pdp_interact_kargs` and `pdp_interact_plot_kargs` to pass to `pdp_interact` and  `pdp_interact_plot`, respectively
- `ForwardHook` class
- `plot_multibody_weighted_outputs` an interpration plot for `MultiBlock` models
- Better documentation for methods and classes

## Removals

- Some default values of arugments in `AMS` and `MultiAMS`
- Default for `return_mean` in `RegAsProxyPull` and `RegPull`

## Fixes

- Missing bbox_inches in `plot_embedding`
- Typing for  `cont_feats` and `savename` in `fit_input_pipe`
- Typing for  `targ_feats` and `savename` in `fit_output_pipe`
- Moved predictions to after callback `on_eval_begin`
- Updated `from_model_builder` class method of `ModelBuilder`to use and `CatEmbedder`
- Hard coded savename in `Model` during save to hopefull solve occaisional permission error during save
- Typing for `val_fold` in `SWA`
- 'lr' to 'momentum' in `Model.set_mom`
- `Model.get_mom` now actually returns momentum (beta_1) rather than lr
- Added catch for infinite uncertainties being passed to `uncert_round`
- Added catch for `plot_roc` with bootstraping when resamples data only contains one class
- Error when attempting to plot categorical feature in `plot_1d_partial_dependence`
- layer width in dense=True `FullyConnected` scaling with input size
- Fixed `lookup_act` for linear function
- `plot_1d_partial_dependence` not using `n_points` parameter
- Errors in `plot_rocs` when passing non-lists and when requesting plot_params and bootsrapping
- Missing `to_device` call when exporting to ONNX on a CUDA device

## Changes

- `to_pt_eta_phi` now infers presence of z momentum from dataframe
- `norm_in` default value for `get_pre_proc_pipes` is now `True` rather than `False`
- `fold_train_ensemble` now always trains `n_models`, and validation fold IDs are cycled through according to `fy.n_folds % model_num`
- `FoldYielder.set_ignore` changed to `FoldYielder.add_ignore`
- Changed `HEPAugFoldYielder.rotate` and `HEPAugFoldYielder.reflect` to private methods
- `compute` method of `RegPull` now private
- Renamed `data` to `fy` in `RegPull.evaluate` and `RegAsProxyPull.evaluate`
- Made `get_layer` in `FullyConnected` private
- Made `get_dense` and `load_embeds` in `CatEmbHead` private
- Made `build_layers` in 'ClassRegMulti` private
- Made parse methods and `build_opt` in `ModelBuilder` private
- Made `get_folds` private
- Changed `settings` to `plot_settings` in `rf_rank_features`
- Dense layer from `CatEmbHead` removed and placed in `FullyConnected`
- Swapped order of continuous and categorical embedding concatination in `CatEmbHead` in order to match input data
- `arr` in `plot_kdes_from_bs` changed to `x`
- weighted partial dependencies in `plot_1d_partial_dependence` are now computed by passing the name of the weight coulmn in the dataframe and normalisation is done automatically
- `data` argument for `plot_binary_class_pred` renamed to `df`
- `plot_1d_partial_dependence` and `plot_2d_partial_dependence` both now expect to be passed a list on training features, rather than expecteing the DataFrame to only contain the trainign features
- rfpimp package nolonger requires manual installation

## Depreciations

- Passing `eta` argument to `to_pt_eta_phi`. Will be removed in v0.4
- `binary_class_cut` renamed to `binary_class_cut_by_ams`. Code added to call `binary_class_cut_by_ams`. Will be removed in v0.4
- `plot_dendrogram` renamed to `plot_rank_order_dendrogram`. Code added to call `plot_rank_order_dendrogram`. Will be removed in v0.4
- `Embedder` renamed to `CatEmbedder`. Code added to call `CatEmbedder`. Will be removed in v0.4
- `n_cont_in` (number of continuous input features) argument of `ModelBuilder` depreciated in favour of `cont_feats` (list of named continuous input features). Code added to create this by encoding numbers as string. Will be removed in v0.4.

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
- Initialisation arguments for `CatEmbHead` changed considerably w.r.t. embedding arguments; now expects to receive a `CatEmbedder` class

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
- Added helper class `CatEmbedder` to simplify parsing of embedding settings
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
    - `callback_args` now depreciated, to be removed in v0.4
    - Currently `callback_args` are converted to `callback_partials`, code will also be removed in v0.4
- Embeddings:
    - Added `cat_embedder` parameter to `ModelBuilder` to eventuall replace `cat_args`
    - `cat_args` now depreciated to be removed in v0.4
    - Currently `cat_args` are converted to an `Embedder`, code will also be removed in v0.4

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
- loss history plot not being saved correctly

## Changes

- Removed the `BatchNorm1d` automatically added in `CatEmbHead` when using categorical inputs; assuming unit-Gaussian continuous inputs, no *a priori* resaon to add it, and tests indicated it hurt performance and train-time.
- Changed weighting factor when not loading loading cycles only to n+2 from n+1

## Depreciations

## Comments

# V0.1.0 PyPI am assuming direct control

Record of changes begins
