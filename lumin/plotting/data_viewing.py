import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple

import scipy
from scipy.cluster import hierarchy as hc

from .plot_settings import PlotSettings
from ..utils.statistics import uncert_round, get_moments

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['plot_feat', 'compare_events', 'plot_rank_order_dendrogram', 'plot_kdes_from_bs', 'plot_binary_sample_feat']


def plot_feat(df:pd.DataFrame, feat:str, wgt_name:Optional[str]=None, cuts:Optional[List[pd.Series]]=None,
              labels:Optional[List[str]]='', plot_bulk:bool=True, n_samples:int=100000,
              plot_params:Optional[Union[Dict[str,Any],List[Dict[str,Any]]]]=None, size:str='mid', show_moments:bool=True,
              ax_labels:Dict[str,Any]={'y': 'Density', 'x': None}, log_x:bool=False, log_y:bool=False, savename:Optional[str]=None,
              settings:PlotSettings=PlotSettings()) -> None:
    r'''
    A flexible function to provide indicative information about the 1D distribution of a feature.
    By default it will produce a weighted KDE+histogram for the [1,99] percentile of the data,
    as well as compute the mean and standard deviation of the data in this region.
    Distributions are weighted by sampling with replacement the data with probabilities propotional to the sample weights.
    By passing a list of cuts and labels, it will plot multiple distributions of the same feature for different cuts.
    Since it is designed to provide quick, indicative information, more specific functions (such as `plot_kdes_from_bs`)
    should be used to provide final results.

    .. important::
        NaN and Inf values are removed prior to plotting and no attempt is made to coerce them to real numbers
    
    Arguments:
        df: Pandas DataFrame containing data
        feat: column name to plot
        wgt_name: if set, will use column to weight data
        cuts: optional list of cuts to apply to feature. Will add one KDE+hist for each cut listed on the same plot
        labels: optional list of labels for each KDE+hist
        plot_bulk: whether to plot the [1,99] percentile of the data, or all of it
        n_samples: if plotting weighted distributions, how many samples to use
        plot_params: optional list of of arguments to pass to Seaborn Distplot for each KDE+hist
        size: string to pass to :meth:`~lumin.plotting.plot_settings.PlotSettings.str2sz` to determin size of plot
        show_moments: whether to compute and display the mean and standard deviation
        ax_labels: dictionary of x and y axes labels
        log_x: if true, will use log scale for x-axis
        log_y: if true, will use log scale for y-axis
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    def _filter_data(x:Union[pd.DataFrame,pd.Series]) -> Union[pd.DataFrame,pd.Series]: return x.replace([np.inf,-np.inf],np.nan).dropna()

    if not isinstance(labels, list): labels = [labels]
    if not isinstance(cuts,   list): cuts   = [cuts]
    if plot_params is None: plot_params = {}
    if len(cuts) != len(labels): raise ValueError(f"{len(cuts)} plots requested, but {len(labels)} labels passed")
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.str2sz(size, 'x'), settings.str2sz(size, 'y')))
        for i in range(len(cuts)):
            tmp_plot_params = plot_params[i] if isinstance(plot_params, list) else plot_params

            if plot_bulk:  # Ignore tails for indicative plotting
                feat_range = np.percentile(_filter_data(df[feat]), [1, 99])
                if feat_range[0] == feat_range[1]: break
                cut = (df[feat] > feat_range[0]) & (df[feat] < feat_range[1])
                if cuts[i] is not None: cut = cut & (cuts[i])
                if wgt_name is None:
                    plot_data = _filter_data(df.loc[cut, feat])
                else:
                    tmp = _filter_data(df.loc[cut, [wgt_name, feat]])
                    weights = tmp[wgt_name].values.astype('float64')
                    weights /= weights.sum()
                    plot_data = np.random.choice(tmp[feat], n_samples, p=weights)
            else:
                tmp_data = df if cuts[i] is None else df.loc[cuts[i]]
                if wgt_name is None:
                    plot_data = _filter_data(tmp_data[feat])
                else:
                    tmp_data = _filter_data(tmp_data[[wgt_name, feat]])
                    weights = tmp_data[wgt_name].values.astype('float64')
                    weights /= weights.sum()
                    plot_data = np.random.choice(tmp_data[feat], n_samples, p=weights)
            label = labels[i]
            if show_moments:
                moms = get_moments(plot_data)
                mean = uncert_round(moms[0], moms[1])
                std  = uncert_round(moms[2], moms[3])
                if wgt_name is None: label += r' $\bar{x}=$' + f'{mean[0]}±{mean[1]}' + r', $\sigma_x=$' + f'{std[0]}±{std[1]}'
                else:                label += r' $\bar{x}=$' + f'{mean[0]}' + r', $\sigma_x=$' + f'{std[0]}'

            sns.distplot(plot_data, label=label, **tmp_plot_params)

        if len(cuts) > 1 or show_moments: plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        if log_y: plt.yscale('log')
        if log_x: plt.xscale('log')
        if log_y or log_x: plt.grid(which="both", axis="y" if log_y and not log_x else "x" if log_x and not log_y else 'both')
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.ylabel(ax_labels['y'], fontsize=settings.lbl_sz, color=settings.lbl_col)
        x_lbl = feat if ax_labels['x'] is None else ax_labels['x']
        plt.xlabel(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def compare_events(events:list) -> None:
    r'''
    Plots at least two events side by side in their transverse and longitudinal projections

    Arguments:
        events: list of DataFrames containing vector coordinates for 3 momenta
    '''

    # TODO: check typing, why list?
    # TODO: make this work with a single event
    # TODO: add plot settings & saving

    with sns.axes_style('whitegrid'), sns.color_palette('tab10'):
        fig, axs = plt.subplots(3, len(events), figsize=(9*len(events), 18), gridspec_kw={'height_ratios': [1, 0.5, 0.5]})
        for vector in [x[:-3] for x in events[0].columns if '_px' in x.lower()]:
            for i, in_data in enumerate(events):
                x = in_data[vector + '_px'].values[0]
                try: y = in_data[vector + '_py'].values[0]
                except KeyError: y = 0
                try: z = in_data[vector + '_pz'].values[0]
                except KeyError: z = 0
                axs[0, i].plot((0, x), (0, y), label=vector)
                axs[1, i].plot((0, z), (0, x), label=vector)
                axs[2, i].plot((0, z), (0, y), label=vector)
        for ax in axs[0]:
            ax.add_artist(plt.Circle((0, 0), 1, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_x$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)  
        for ax in axs[1]:
            ax.add_artist(plt.Rectangle((-2, -1), 4, 2, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_x$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        for ax in axs[2]: 
            ax.add_artist(plt.Rectangle((-2, -1), 4, 2, color='grey', fill=False, linewidth=2))
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(r"$p_z$", fontsize=16, color='black')
            ax.set_ylabel(r"$p_y$", fontsize=16, color='black')
            ax.legend(loc='right', fontsize=12)
        fig.show()


def plot_rank_order_dendrogram(df:pd.DataFrame, threshold:float=0.8, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) \
        -> Dict[str,Union[List[str],float]]:
    r'''
    Plots a dendrogram of features in df clustered via Spearman's rank correlation coefficient.
    Also returns a sets of features with correlation coefficients greater than the threshold

    Arguments:
        df: Pandas DataFrame containing data
        threshold: Threshold on correlation coefficient
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance

    Returns:
        Dict of sets of features with correlation coefficients greater than the threshold and cluster distance
    '''

    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-np.abs(corr))  # Abs because negtaive of a feature is a trvial transformation: information unaffected
    z = hc.linkage(corr_condensed, method='average', optimal_ordering=True)

    with sns.axes_style('white'), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_large, (0.5*len(df.columns))))
        hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=settings.lbl_sz, color_threshold=1-threshold)
        plt.xlabel("Distance (1 - |Spearman's Rank Correlation Coefficient|)", fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()

    feats = df.columns
    sets = {}
    for i, merge in enumerate(z):
        if merge[2] > 1-threshold: continue
        if merge[0] <= len(z): a = [feats[int(merge[0])]]
        else:                  a = sets.pop(int(merge[0]))['children']
        if merge[1] <= len(z): b = [feats[int(merge[1])]]
        else:                  b = sets.pop(int(merge[1]))['children']
        sets[1 + i + len(z)] = {'children': [*a, *b], 'distance': merge[2]}
    return sets


def plot_kdes_from_bs(x:np.ndarray, bs_stats:Dict[str,Any], name2args:Dict[str,Dict[str,Any]], 
                      feat:str, units:Optional[str]=None, moments=True,
                      savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots KDEs computed via :meth:`~lumin.utils.statistics.bootstrap_stats`

    Arguments:
        bs_stats: (filtered) dictionary retruned by :meth:`~lumin.utils.statistics.bootstrap_stats`
        name2args: Dictionary mapping names of different distributions to arguments to pass to seaborn tsplot
        feat: Name of feature being plotted (for axis lablels)
        units: Optional units to show on axes
        moments: whether to display mean and standard deviation of each distribution
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''

    # TODO: update to sns 9

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i, name in enumerate(name2args):
            if 'color' not in name2args[name]: name2args[name]['color'] = palette[i]
            if 'label' in name2args[name]:
                name2args[name]['condition'] = name2args[name]['label']
                name2args[name].pop('label')
            if 'condition' in name2args[name] and moments:
                mean, mean_unc = uncert_round(np.mean(bs_stats[f'{name}_mean']), np.std(bs_stats[f'{name}_mean'], ddof=1))
                std, std_unc = uncert_round(np.mean(bs_stats[f'{name}_std']), np.std(bs_stats[f'{name}_std'], ddof=1))
                name2args[name]['condition'] += r', $\overline{x}=' + r'{}\pm{}\ \sigma= {}\pm{}$'.format(mean, mean_unc, std, std_unc)
            sns.tsplot(data=bs_stats[f'{name}_kde'], time=x, **name2args[name])

        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        y_lbl = r'$\frac{1}{N}\ \frac{dN}{d' + feat.replace('$','') + r'}$'
        if units is not None:
            x_lbl = feat + r'$\ [' + units + r']$'
            y_lbl += r'$\ [' + units + r'^{-1}]$'
        else:
            x_lbl = feat
        plt.xlabel(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(y_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_binary_sample_feat(df:pd.DataFrame, feat:str, targ_name:str='gen_target', wgt_name:str='gen_weight', sample_name:str='gen_sample',
                            wgt_scale:float=1, bins:Optional[Union[int,List[int]]]=None, log_y:bool=False, lim_x:Optional[Tuple[float,float]]=None,
                            density=True, feat_name:Optional[str]=None, units:Optional[str]=None,
                            savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    More advanced plotter for feature distributions in a binary class problem with stacked distributions for backgrounds and user-defined binning
    Note that plotting colours can be controled by seeting the settings.sample2col dictionary
    
    Arguments:
        df: DataFrame with targets and predictions
        feat: name of column to plot the distribution of
        targ_name: name of column to use as targets
        wgt_name: name of column to use as sample weights
        sample_name: name of column to use as process names
        wgt_scale: applies a global multiplicative rescaling to sample weights. Default 1 = no rescaling. Only applicable when density = False     
        bins: either the number of bins to use for a uniform binning, or a list of bin edges for a variable-width binning
        log_y: whether to use a log scale for the y-axis
        lim_x: limit for plotting on the x-axis
        density: whether to normalise each distribution to one, or keep set to sum of weights / datapoints
        feat_name: Name of feature to put on x-axis, can be in LaTeX.
        units: units used to measure feature, if applicable. Can be in LaTeX, but should not include '$'.
        savename: Optional name of file to which to save the plot of feature importances
        settings: :class:`~lumin.plotting.plot_settings.PlotSettings` class to control figure appearance
    '''
    
    def _get_samples(df:pd.DataFrame, sample_name:str, wgt_name:str):
        '''Returns set of samples present in df ordered by sum of weights''' 
        samples = set(df[sample_name])
        weights = [np.sum(df[df[sample_name] == sample][wgt_name]) for sample in samples]
        return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))]
    
    sig,bkg = (df[targ_name] == 1),(df[targ_name] == 0)
    if not isinstance(bins,list): bins = np.linspace(df[feat].min(),df[feat].max(), bins if isinstance(bins, int) else 20)
    hist_params = {'range': lim_x, 'bins': bins, 'density': density, 'alpha': 0.8, 'stacked':True, 'rwidth':1.0}
    sig_samples = _get_samples(df[sig], sample_name, wgt_name)
    bkg_samples = _get_samples(df[bkg], sample_name, wgt_name)
    sample2col = {k: v for v, k in enumerate(bkg_samples)} if settings.sample2col is None else settings.sample2col
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette, 1+max([sample2col[x] for x in sample2col])):
        fig, ax = plt.subplots(figsize=(settings.w_mid, settings.h_mid))
        ax.hist([df[df[sample_name] == sample][feat] for sample in bkg_samples],
                weights=[wgt_scale*df[df[sample_name] == sample][wgt_name] for sample in bkg_samples],
                label=bkg_samples, color=[sns.color_palette()[sample2col[s]] for s in bkg_samples], **hist_params)
        
        for sample in sig_samples:
            ax.hist(df[df[sample_name] == sample][feat],
                    weights=wgt_scale*df[df[sample_name] == sample][wgt_name],
                    label=sample, histtype='step', linewidth='3', 
                    color='black', **hist_params)
        
        ax.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        if lim_x is not None: ax.set_xlim(*lim_x)
        ax.tick_params(axis='x', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
        ax.tick_params(axis='y', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
        x_lbl = feat if feat_name is None else feat_name
        y_lbl = r'$\frac{d\left(\mathcal{A}\sigma\right)}{dx}$'
        if units is not None:
            x_lbl += r'$\ [' + units + r']$'
            y_lbl += r'$\ [' + units + r'^{-1}]$'
        ax.xaxis.set_label_text(x_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        if density: ax.yaxis.set_label_text(r"$\frac{1}{\mathcal{A}\sigma}$"+y_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        else:       ax.yaxis.set_label_text(r"$\mathcal{L}_{\mathrm{int.}}\times$"+y_lbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        if log_y:
            ax.set_yscale('log', nonposy='clip')
            ax.grid(True, which="both")
        ax.set_title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        fig.show()
        