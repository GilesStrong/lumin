
from typing import Optional
import pandas as pd

from .plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt


def plot_fi(df:pd.DataFrame, feat_name:str='Feature', imp_name:str='Importance',  unc_name:str='Uncertainty',
            savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    with sns.axes_style(settings.style), sns.color_palette(settings.palette):
        fig, ax = plt.subplots(figsize=(settings.w_large, (0.5)*settings.lbl_sz))
        xerr = None if unc_name not in df else 'Uncertainty'
        df.plot(feat_name, imp_name, 'barh', ax=ax, legend=False, xerr=xerr, error_kw={'elinewidth': 3})
        ax.set_xlabel('Importance via feature permutation', fontsize=16, color='black')
        ax.set_ylabel('Feature', fontsize=16, color='black')
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}')
        plt.show()
