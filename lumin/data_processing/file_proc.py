import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

__all__ = ["save_to_grp", "fold2foldfile", "df2foldfile", "add_meta_data"]


def save_to_grp(arr: np.ndarray, grp: h5py.Group, name: str, compression: Optional[str] = None) -> None:
    r"""
    Save Numpy array as a dataset in an h5py Group

    Arguments:
        arr: array to be saved
        grp: group in which to save arr
        name: name of dataset to create
        compression: optional compression argument for h5py, e.g. 'lzf'
    """

    # TODO Option for string length

    grp.create_dataset(
        name,
        shape=arr.shape,
        dtype=arr.dtype.name if arr.dtype.name not in ["object", "str864"] else "S64",
        data=arr if arr.dtype.name not in ["object", "str864"] else arr.astype("S64"),
        compression=compression,
    )


def _build_matrix_lookups(
    feats: List[str], vecs: List[str], feats_per_vec: List[str], row_wise: bool
) -> Tuple[List[str], np.ndarray, Tuple[int, int]]:
    shape = (len(vecs), len(feats_per_vec)) if row_wise else (len(feats_per_vec), len(vecs))
    lookup, missing = np.zeros(shape, dtype=np.array(feats).dtype), np.zeros(shape, dtype=bool)
    if row_wise:
        for i, v in enumerate(vecs):
            for j, c in enumerate(feats_per_vec):
                f = f"{v}_{c}"
                if f in feats:
                    lookup[i, j] = f
                else:
                    lookup[i, j] = feats[0]  # Temp value, to be set to null later using missing
                    missing[i, j] = True
    else:
        for j, v in enumerate(vecs):
            for i, c in enumerate(feats_per_vec):
                f = f"{v}_{c}"
                if f in feats:
                    lookup[i, j] = f
                else:
                    lookup[i, j] = feats[0]  # Temp value, to be set to null later using missing
                    missing[i, j] = True
    return list(lookup.flatten()), missing.flatten(), shape


def fold2foldfile(
    df: Optional[pd.DataFrame],
    out_file: h5py.File,
    fold_idx: int,
    cont_feats: List[str],
    cat_feats: List[str],
    targ_feats: Union[str, List[str]],
    targ_type: Any,
    misc_feats: Optional[List[str]] = None,
    wgt_feat: Optional[str] = None,
    matrix_lookup: Optional[List[str]] = None,
    matrix_missing: Optional[np.ndarray] = None,
    matrix_shape: Optional[Tuple[int, int]] = None,
    tensor_data: Optional[np.ndarray] = None,
    tensor_target: Optional[np.ndarray] = None,
    compression: Optional[str] = None,
    n_samples: Optional[int] = None,
) -> None:
    r"""
    Save fold of data into an h5py Group

    Arguments:
        df: Dataframe from which to save data, can contain flat input features, weights, targets,
            but is entriely optional if targets and inputs are passed as tensors
        out_file: h5py file to save data in
        fold_idx: ID for the fold; used name h5py group according to 'fold_{fold_idx}'
        cont_feats: list of columns in df to save as continuous variables
        cat_feats: list of columns in df to save as discreet variables
        targ_feats: (list of) column(s) in df to save as target feature(s)
        targ_type: type of target feature, e.g. int,'float32'
        misc_feats: any extra columns to save
        wgt_feat: column to save as data weights
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
        tensor_data: data of higher order than a matrix can be passed directly as a numpy array, rather than beign extracted and reshaped from the DataFrame.
            The array will be saved under matrix data, and this is incompatible with also setting `matrix_lookup`, `matrix_missing`, and `matrix_shape`.
            The first dimension of the array must be compatible with the length of the data frame.
        tensor_target: optional encoding of multi-dimensional targets as a numpy array
        compression: optional compression argument for h5py, e.g. 'lzf'
        n_samples: in case df is None, please supply the number of samples in the fold: this cannot be determined otherwise, since tensor_data and tensor_target may be sparse
    """

    # TODO infer target type automatically

    if df is None and n_samples is None:
        raise ValueError(
            "Please supply the number of samples in the fold (n_samples): this cannot be determined otherwise, since tensor_data and tensor_target may be sparse"
        )

    grp = out_file.create_group(f"fold_{fold_idx}")

    if df is not None:
        save_to_grp(
            np.hstack((df[cont_feats].values.astype("float32"), df[cat_feats].values.astype("float32"))),
            grp,
            "inputs",
            compression=compression,
        )
    else:
        save_to_grp(
            np.empty((n_samples, 0)), grp, "inputs", compression=compression
        )  # Inputs must exist for compatibility and data-count reasons
    if tensor_target is None:
        save_to_grp(df[targ_feats].values.astype(targ_type), grp, "targets", compression=compression)
    else:
        save_to_grp(tensor_target.astype(targ_type), grp, "targets", compression=compression)
    if wgt_feat is not None:
        if wgt_feat in df.columns:
            save_to_grp(df[wgt_feat].values.astype("float32"), grp, "weights", compression=compression)
        else:
            print(f"{wgt_feat} not found in file")
    if misc_feats is not None:
        for f in misc_feats:
            if f in df.columns:
                save_to_grp(df[f].values, grp, f, compression=compression)
            else:
                print(f"{f} not found in file")

    if matrix_lookup is not None:
        if tensor_data is not None:
            raise ValueError(
                "The saving of both matrix and tensor data is requested. This is ambiguous. Please only set one of the other."
            )
        mat = df[matrix_lookup].values.astype("float32")
        mat[:, matrix_missing] = np.NaN
        mat = mat.reshape((len(df), *matrix_shape))
        save_to_grp(mat, grp, "matrix_inputs", compression=compression)

    elif tensor_data is not None:
        save_to_grp(tensor_data.astype("float32"), grp, "matrix_inputs", compression=compression)


def df2foldfile(
    df: Optional[pd.DataFrame],
    n_folds: int,
    cont_feats: List[str],
    cat_feats: List[str],
    targ_feats: Union[str, List[str]],
    savename: Union[Path, str],
    targ_type: str,
    shuffle: bool = True,
    strat_key: Optional[str] = None,
    misc_feats: Optional[List[str]] = None,
    wgt_feat: Optional[str] = None,
    cat_maps: Optional[Dict[str, Dict[int, Any]]] = None,
    matrix_vecs: Optional[List[str]] = None,
    matrix_feats_per_vec: Optional[List[str]] = None,
    matrix_row_wise: Optional[bool] = None,
    tensor_data: Optional[np.ndarray] = None,
    tensor_name: Optional[str] = None,
    tensor_as_sparse: bool = False,
    compression: Optional[str] = None,
    tensor_target: Optional[np.ndarray] = None,
    tensor_target_as_sparse: bool = False,
) -> None:
    r"""
    Convert dataframe into h5py file by splitting data into sub-folds to be accessed by a :class:`~lumin.nn.data.fold_yielder.FoldYielder`

    Arguments:
        df: Dataframe from which to save data, can contain flat input features, weights, targets,
            but is entriely optional if targets and inputs are passed as tensors
        n_folds: number of folds to split df into
        cont_feats: list of columns in df to save as continuous variables
        cat_feats: list of columns in df to save as discreet variables
        targ_feats: (list of) column(s) in df to save as target feature(s)
        savename: name of h5py file to create (.h5py extension not required)
        targ_type: type of target feature, e.g. int,'float32'
        shuffle: if true will shuffle data prior to splitting into folds, otherwise folds will be contiguous splits of the unsuffled data, useful e.g. for testing datasets
        strat_key: column to use for stratified splitting
        misc_feats: any extra columns to save
        wgt_feat: column to save as data weights
        cat_maps: Dictionary mapping categorical features to dictionary mapping codes to categories
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
        tensor_data: data of higher order than a matrix can be passed directly as a numpy array, rather than beign extracted and reshaped from the DataFrame.
            The array will be saved under matrix data, and this is incompatible with also setting `matrix_vecs`, `matrix_feats_per_vec`, and `matrix_row_wise`.
            The first dimension of the array must be compatible with the length of the data frame.
        tensor_name: if `tensor_data` is set, then this is the name that will to the foldfile's metadata.
        tensor_as_sparse: Set to True to store the matrix in sparse COO format
            The format is `coo_x = sparse.as_coo(x); m = np.vstack((coo_x.data, coo_x.coords))`, where `m` is the tensor passed to `tensor_data`.
        compression: optional compression argument for h5py, e.g. 'lzf'
        tensor_target: optional encoding of multi-dimensional targets as a numpy array
        tensor_target_as_sparse:  Set to True to store the matrix in sparse COO format
            The format is `coo_x = sparse.as_coo(x); m = np.vstack((coo_x.data, coo_x.coords))`, where `m` is the tensor passed to `tensor_target`.
    """

    savename = str(savename)
    if shuffle and ("test" in savename or "tst" in savename):
        print("Testing data will be shuffled, pass shuffle=Flase is this is not desired")
    os.system(f"rm {savename}.hdf5")
    os.makedirs(savename[: savename.rfind("/")], exist_ok=True)
    out_file = h5py.File(f"{savename}.hdf5", "w")
    lookup, missing, shape = None, None, None
    if matrix_vecs is not None:
        if tensor_data is not None:
            raise ValueError(
                "The saving of both matrix and tensor data is requested. This is ambiguous. Please only set one of the other."
            )
        lookup, missing, shape = _build_matrix_lookups(df.columns, matrix_vecs, matrix_feats_per_vec, matrix_row_wise)
        mat_feats = list(np.array(lookup)[np.logical_not(missing)])  # Only features present in data
        dup = [f for f in cont_feats if f in mat_feats]
        if len(dup) > 1:
            print(f"{dup} present in both matrix features and continuous features; removing from continuous features")
            cont_feats = [f for f in cont_feats if f not in dup]

    if strat_key is not None and strat_key not in df.columns:
        print(f"{strat_key} not found in DataFrame")
        strat_key = None
    if strat_key is None or shuffle is False:
        kf = KFold(n_splits=n_folds, shuffle=shuffle)
        folds = kf.split(X=df if df is not None else tensor_data)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = kf.split(X=df, y=df[strat_key])

    if tensor_as_sparse or tensor_target_as_sparse:
        import sparse
    for fold_idx, (_, fold) in enumerate(folds):
        print(f"Saving fold {fold_idx} with {len(fold)} events")

        if tensor_data is not None:
            fold_tensor_data = tensor_data[fold]
            if tensor_as_sparse:
                fold_tensor_data = sparse.as_coo(fold_tensor_data)
                fold_tensor_data = np.vstack((fold_tensor_data.data, fold_tensor_data.coords))
        if tensor_target is not None:
            fold_tensor_targ = tensor_target[fold]
            if tensor_target_as_sparse:
                fold_tensor_targ = sparse.as_coo(fold_tensor_targ)
                fold_tensor_targ = np.vstack((fold_tensor_targ.data, fold_tensor_targ.coords))

        fold2foldfile(
            df.iloc[fold].copy() if df is not None else None,
            out_file,
            fold_idx,
            cont_feats=cont_feats,
            cat_feats=cat_feats,
            targ_feats=targ_feats,
            targ_type=targ_type,
            misc_feats=misc_feats,
            wgt_feat=wgt_feat,
            matrix_lookup=lookup,
            matrix_missing=missing,
            matrix_shape=shape,
            tensor_data=fold_tensor_data if tensor_data is not None else None,
            tensor_target=fold_tensor_targ if tensor_target is not None else None,
            compression=compression,
            n_samples=len(fold),
        )
    add_meta_data(
        out_file=out_file,
        feats=df.columns if df is not None else [],
        cont_feats=cont_feats,
        cat_feats=cat_feats,
        cat_maps=cat_maps,
        targ_feats=targ_feats,
        wgt_feat=wgt_feat,
        matrix_vecs=matrix_vecs,
        matrix_feats_per_vec=matrix_feats_per_vec,
        matrix_row_wise=matrix_row_wise,
        tensor_name=tensor_name,
        tensor_shp=tensor_data[0].shape if tensor_data is not None else None,
        tensor_is_sparse=tensor_as_sparse,
        target_tensor_shp=tensor_target[0].shape if tensor_target is not None else None,
        tensor_target_is_sparse=tensor_target_as_sparse,
    )


def add_meta_data(
    out_file: h5py.File,
    feats: List[str],
    cont_feats: List[str],
    cat_feats: List[str],
    cat_maps: Optional[Dict[str, Dict[int, Any]]],
    targ_feats: Union[str, List[str]],
    wgt_feat: Optional[str] = None,
    matrix_vecs: Optional[List[str]] = None,
    matrix_feats_per_vec: Optional[List[str]] = None,
    matrix_row_wise: Optional[bool] = None,
    tensor_name: Optional[str] = None,
    tensor_shp: Optional[Tuple[int]] = None,
    tensor_is_sparse: bool = False,
    target_tensor_shp: Optional[Tuple[int]] = None,
    tensor_target_is_sparse: bool = False,
) -> None:
    r"""
    Adds meta data to foldfile containing information about the data: feature names, matrix information, etc.
    :class:`~lumin.nn.data.fold_yielder.FoldYielder` objects will access this and automatically extract it to save the user from having to manually pass lists
    of features.

    Arguments:
        out_file: h5py file to save data in
        feats: list of all features in data
        cont_feats: list of continuous features
        cat_feats: list of categorical features
        cat_maps: Dictionary mapping categorical features to dictionary mapping codes to categories
        targ_feats: (list of) target feature(s)
        wgt_feat: name of weight feature
        matrix_vecs: list of objects for matrix encoding, i.e. feature prefixes
        matrix_feats_per_vec: list of features per vector for matrix encoding, i.e. feature suffixes.
            Features listed but not present in df will be replaced with NaN.
        matrix_row_wise: whether objects encoded as a matrix should be encoded row-wise (i.e. all the features associated with an object are in their own row),
            or column-wise (i.e. all the features associated with an object are in their own column)
        tensor_name: Name used to refer to the tensor when displaying model information
        tensor_shp: The shape of the tensor data (excluding batch dimension)
        tensor_is_sparse: Whether the tensor is sparse (COO format) and should be densified prior to use
        target_tensor_shp: The shape of the target tensor data (excluding batch dimension)
        tensor_target_is_sparse: Whether the target tensor is sparse (COO format) and should be densified prior to use
    """

    grp = out_file.create_group("meta_data")
    grp.create_dataset("cont_feats", data=json.dumps(cont_feats))
    grp.create_dataset("cat_feats", data=json.dumps(cat_feats))
    grp.create_dataset("targ_feats", data=json.dumps(targ_feats))
    if wgt_feat is not None:
        grp.create_dataset("wgt_feat", data=json.dumps(wgt_feat))
    if cat_maps is not None:
        grp.create_dataset("cat_maps", data=json.dumps(cat_maps))
    if matrix_vecs is not None:
        lookup, missing, shape = _build_matrix_lookups(feats, matrix_vecs, matrix_feats_per_vec, matrix_row_wise)
        use = list(np.array(lookup)[np.logical_not(missing)])  # Only features present in data
        grp.create_dataset(
            "matrix_feats",
            data=json.dumps(
                {
                    "present_feats": use,
                    "vecs": matrix_vecs,
                    "missing": [int(m) for m in missing],
                    "feats_per_vec": matrix_feats_per_vec,
                    "row_wise": matrix_row_wise,
                    "shape": shape,
                }
            ),
        )
    elif tensor_name is not None:
        grp.create_dataset(
            "matrix_feats",
            data=json.dumps(
                {
                    "present_feats": [tensor_name],
                    "vecs": [tensor_name],
                    "missing": [],
                    "feats_per_vec": [""],
                    "row_wise": None,
                    "shape": tensor_shp,
                    "is_sparse": tensor_is_sparse,
                }
            ),
        )
    if target_tensor_shp is not None:
        grp.create_dataset(
            "target_tensor", data=json.dumps({"shape": target_tensor_shp, "is_sparse": tensor_target_is_sparse})
        )
