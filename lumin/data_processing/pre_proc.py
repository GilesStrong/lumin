from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def get_pre_proc_pipes(norm_in=False, norm_out=False, pca=False, whiten=False, with_mean=True, with_std=True):
    steps_in = []
    if not norm_in and not pca:
        steps_in.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    else:
        if pca: steps_in.append(('pca', PCA(whiten=whiten)))
        if norm_in: steps_in.append(('norm_in', StandardScaler(with_mean=with_mean, with_std=with_std)))
    input_pipe = Pipeline(steps_in)

    steps_out = []
    if norm_out: steps_out.append(('norm_out', StandardScaler(with_mean=with_mean, with_std=with_std)))
    else:        steps_out.append(('ident', StandardScaler(with_mean=False, with_std=False)))  # For compatability
    output_pipe = Pipeline(steps_out)
    return input_pipe, output_pipe
