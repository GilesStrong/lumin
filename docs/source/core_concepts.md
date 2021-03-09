## The fold file

The fold file is the core data-structure used throughout LUMIN. It is stored on disc as an HDF5 file. In the top level are several groups. The `meta_data` group stores various datasets containing information about the data, such as the names of features. The other top-level groups are the *folds*. These store subsamples of the full dataset and are designed to be read into memory individually, and provide several advantages, such as:

- Memory requirements are reduced
- Specific fold indices can be designated for training and others for validation, e.g. for k-fold cross-validation
- Some methods can compute averaged metrics over folds and produce uncertainties based on standard deviation

Each fold group contains several datasets:

- `targets` will be used to provide target data for training NNs
- `inputs` contains the input data in the following
- `weights`, if present, will be used to weight losses during training
- `matrix_inputs` can be used to store 2D matrix, or higher-order (sparse) tensor data

Additional datasets can be added, too, e.g. extra features that are necessary for interpreting results. Named predictions can also be saved to the fold file e.g. during `Model.predict`. Datasets can also be compressed to reduce size and loading time.

### Creating fold files

`lumin.data_processing.file_proc` contains the recommended methods for constructing fold files from `pandas.DataFrame` objects with the main construction method being [df2foldfile](https://lumin.readthedocs.io/en/stable/lumin.data_processing.html#lumin.data_processing.file_proc.df2foldfile), although the methods it calls can be used directly for complex or large data.

### Reading fold files

The main interface class is the [FoldYielder](https://lumin.readthedocs.io/en/stable/lumin.nn.data.html#lumin.nn.data.fold_yielder.FoldYielder). Its primary function is to load data from the fold files, however it can also act as hub for meta information and objects concerning the dataset, such as feature names and processing pipelines. Specific features can be marked as 'ignore' and will be filtered out when loading folds.

Calling `fy.get_fold(i)` or indexing an instance `fy[i]` will return a dictionary of inputs, targets, and weights for fold `i` via the `get_data` method. Flat inputs will be passed through `np.nan_to_num`. If matrix or tensor inputs are present then they will be processed into a tuple with the flat data ([flat inputs, dense tensor]).

`fy.get_df` can be used to construct a `pandas.DataFrame` from the data (either specific folds, or all folds together). The method has various arguments for controlling what columns should be included. By default only targets, weights, and predictions are included. Additional datasets can also be loaded via the `get_column` method.

Since during training and inference folds are loaded into memory one at a time, used once, and overwritten LUMIN can optionally apply data augmentation when loading folds. The inheriting class [HEPAugFoldYielder](https://lumin.readthedocs.io/en/stable/lumin.nn.data.html#lumin.nn.data.fold_yielder.HEPAugFoldYielder) provides an example of this, where particle collision events can be rotated and flipped.

## Models

### Model building

Contrary to other high-level interfaces, in LUMIN the user is expected to define **how** models, optimisers, and loss functions should be built, rather than build them themselves. The [ModelBuilder](https://lumin.readthedocs.io/en/stable/lumin.nn.models.html#lumin.nn.models.model_builder.ModelBuilder) class helps to capture these definitions, and once instantiated, can be used produce models on demand.

LUMIN models consist of three types of *blocks*:

1. The Head, which takes all inputs from the data and processes them if necessary.
    - The default head is [CatEmbHead](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#module-lumin.nn.models.blocks.head), which passes continuous inputs through an optional dropout layer, and categorical inputs through embedding matrices (see [Guo & Berkhahn, 2016](https://arxiv.org/abs/1604.06737)) and an optional dropout layer.
    - Matrix or tensor data can also be passed through appropriate head blocks, e.g. [RNNs](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.head.RecurrentHead), [CNNs](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.head.AbsConv1dHead), and [GNNs](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.head.InteractionNet).
    - Data containing both matrix/tensor data and flat data (continuous+categorical) can be passed through a [MultiHead](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.head.MultiHead) block, which in turn sends data through the appropriate head and concatenates the outputs.
    - The output of the head is a flat vector (batch, head width)
1. The Body is where the majority of the computation occurs (at least in the case of a flat FCNN). The default body is [FullyConnected](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.body.FullyConnected), consisting of multiple hidden layers.
    - [MultiBlock](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.body.MultiBlock) can be used to split features across separate body blocks, e.g. for [wide-deep networks](https://arxiv.org/abs/1606.07792).
    - The output of the body is also a flat vector (batch, body width)
1. The Tail is designed to alter the body width to match the target width, as well as apply any output activation function and output rescaling.
    - The default tail is [ClassRegMulti](https://lumin.readthedocs.io/en/stable/lumin.nn.models.blocks.html#lumin.nn.models.blocks.tail.ClassRegMulti), which can handle single- & multi-target regression, and binary, multi-label, and multi-class classification (it configures itself using the `objective` attribute of the `ModelBuilder`).

The `ModelBuilder` has arguments to change the blocks from their default values. Custom blocks should be passed as classes, rather than instantiated objects (i.e. use `partial` to configure their arguments). There are some arguments for blocks which will be set automatically by the `ModelBuilder`: For heads, these are `cont_feats`, `cat_embedder`, `lookup_init`, and `freeze`; for bodies `n_in`, `feat_map`, `lookup_init`, `lookup_act`, and `freeze`; and for tails `n_in`, `n_out`, `objective`, `lookup_init`, and `freeze`. `model_args` can also be used to set arguments via a dictionary, e.g. `{'head':{'depth':3}}`.

The `ModelBuilder` also returns an optimiser set to update the parameters of the model. This can be configured via `opt_args` and custom optimisers can be passed as classes to `'opt'`, e.g. `opt_args={'opt':AdamW, 'lr':3e-2}`. The loss function is controlled by the `loss` argument, and can either be left as auto and set via the `objective`, or explicitly set to a class by the user. Use of pretrained models can be achieved by setting the `pretrain_file` argument to a previously trained model, which will then be loaded when a new model is built.

### Model wrapper

The models built by the `ModelBuilder` are `torch.nn.Module` objects, and so to provide high-level functionality, LUMIN wraps these objects with a [Model](https://lumin.readthedocs.io/en/stable/lumin.nn.models.html#lumin.nn.models.model.Model) class. This provides a range of methods for e.g. training, saving, loading, and predicting with DNNs. The `torch.nn.Module` is set as the `Model.model` attribute.

A similar high-level wrapper class exists for ensembles ([Ensemble](https://lumin.readthedocs.io/en/stable/lumin.nn.ensemble.html#lumin.nn.ensemble.ensemble.Ensemble)), in which the methods extend over a range of `Model` objects.

## Model training

`Model.fit` will train `Model.model` using data provided via a `FoldYielder`. A specific fold index can be set to be used as validation data, and the rest will be used as training data (or the user can specify explicitly which fold indices to use for training). Callbacks can be used to augment the training, as described later on. Training is 'stateful', with a `Model.fit_params` object having various attributes such as the data, current state (training or validation), and callbacks. Since each callback has the model as an attribute, they can access all aspects of the training via the `fit_params`.

Training proceeds thusly:

1. For epoch in epochs:
    1. Training epoch begins
        1. Training-fold indices are shuffled
        1. For fold in training folds (referred to as a *sub-epoch*):
            1. Load fold data into a [BatchYielder](https://lumin.readthedocs.io/en/stable/lumin.nn.data.html#lumin.nn.data.batch_yielder.BatchYielder), a class that yields batches of input, target, and weight data
            1. For batch in batches:
                1. Pass inputs `x` through network to get predictions `y_pred`
                1. Compute loss based on `y_pred` and targets `y`
                1. Back-propagate loss through network
                1. Update network parameters using optimiser
    1. Validation epoch begins
        1. Load validation-fold data into a `BatchYielder`
            1. For batch in batches:
                1. Pass inputs `x` through network to get predictions `y_pred`
                1. Compute loss based on `y_pred` and targets `y`

### Training method

Whilst `Model.fit` can be used by the user, there is still a lot of boilerplate code that must be written to support convenient training and monitoring of models, plus one of the distinguishing characteristics of LUMIN is that training many models should be as easy as training one model. To this end, the recommended training function is [train_models](https://lumin.readthedocs.io/en/stable/lumin.nn.training.html#lumin.nn.training.train.train_models). This function will train a specified number of models and save them to a specified directory. It doesn't return the trained models, but rather a dictionary of results containing information about the training, and the paths to the models. This can then be used to instantiate an [Ensemble](https://lumin.readthedocs.io/en/stable/lumin.nn.ensemble.html#lumin.nn.ensemble.ensemble.Ensemble) via the `from_results` class-method.

## Callbacks

Just like in Keras and FastAI, callbacks are a powerful and flexible way to augment the general training loop outlined above, by offering series of fine-grained interjection points:

- on_train_begin: after all preparations are made and the first epoch is about to begin; allows callbacks to initialise and prepare for the training
- on_epoch_begin: when a new training or validation epoch is about to begin
- on_fold_begin: when a new training or validation fold is about to begin and after the batch yielder has been instantiated; allows callbacks to modify the entirety of the data for the fold via `fit_params.by`
- on_batch_begin: when a new batch or data is about to be processed and inputs, targets, and weights have been set to `fit_params.x`,  fit_params.y`,  and fit_params.w`; allows callbacks to modify the batch before it is passed through the network
- on_forwards_end: after the inputs have been passed through the network and the predictions `fit_params.y_pred` and the loss value `fit_params.loss_val` computed; allows callbacks to modify the loss before it is back-propagated (e.g. adversarial training), or to compute a new loss value and set `fit_params.loss_val` manually
- on_backwards_begin: after the optimiser gradients have been zeroed and before the loss value has been back-propagated
- on_backwards_end: after the loss value has been back-propagated but before the optimiser update has been made; allows callbacks to modify the parameter gradients
- on_batch_end: after the batch has been processed, the loss computed, and any parameter updates made
- on_fold_end: after a training or validation fold has finished
- on_epoch_end: after a training or validation epoch has finished
- on_train_end: after the training has finished; allows callbacks to clean up and compute final results

In addition to callbacks during training, LUMIN offers callbacks at prediction, which can interject at:

- on_pred_begin: After all preparations are made and the prediction data has been loaded into a `BatchYielder`
- on_batch_begin
- on_forwards_end
- on_batch_end
- on_pred_end: After predictions have been made for al the data

Callbacks passed to the `Model` prediction methods come in two varieties: normal callbacks can be passed to `cbs`; and a special *prediction callback* can be passed to `pred_cb`. The prediction callback is responsible for storing and processing model predictions, and then returning the via a `get_preds` method. The default prediction callback simply returns predictions in the same order they were generated, however users may wish to e.g. rescale or bin predictions for convenience. An example use for other callbacks during prediction would be e.g. for inference of parameterised training model [ParameterisedPrediction](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.data_callbacks.ParametrisedPrediction), [Baldi et al., 2016](https://arxiv.org/abs/1601.07913).

### Callbacks in LUMIN

A range of common, or useful, callbacks are provided in LUMIN:

- [Optimiser](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.opt_callbacks) and [Cyclic callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.cyclic_callbacks) are designed to modify optimiser hyper-parameters during training, e.g. [OneCycle](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.cyclic_callbacks.OneCycle) [Smith, 2018](https://arxiv.org/abs/1803.09820). Classes inheriting from [AbsCyclicCallback](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.cyclic_callbacks.AbsCyclicCallback) can signal to other callbacks to only act when a cycle has finished (e.g. stop training after no improvement).
- [Data callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.data_callbacks) modify aspects of the data, e.g. for label smoothing, resampling, and replacing/removing values and data.
- [Loss callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.loss_callbacks) adjust the loss values and gradients, or even manually compute losses themselves.
- [Model callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.model_callbacks) are a special type of callback that trains alternative models and can be polled for loss values, have their performance tracked, and have their models saved instead of the main model, e.g. [SWA](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.model_callbacks.SWA) [Izmailov et al., 2018](https://arxiv.org/abs/1803.05407).
- [Monitor callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.monitors) keep [track of performance during the training](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.monitors.MetricLogger), and provide a realtime report of metrics. Additionally, they can be used to [save models when performance improves](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.monitors.SaveBest) and [stop training after improvements cease](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#lumin.nn.callbacks.monitors.EarlyStopping).
- [Prediction handler callbacks](https://lumin.readthedocs.io/en/stable/lumin.nn.callbacks.html#module-lumin.nn.callbacks.pred_handlers) are responsible for storing and adjusting the network outputs when predicting on new data.
