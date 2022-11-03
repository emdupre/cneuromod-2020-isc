import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nilearn.maskers import NiftiMasker

"""
I would use the subject as primary mode of sorting,
as this seems to be driving the similarities
(as expected based on the Gratton paper).
I would add the repetitions of life and figures in this matrix. 
"""


tasks = ['bourne', 'figures_run-1', 'life_run-1',
         'figures_run-2', 'life_run-2', 'wolf']


def plot_corr_mtx(task, data_dir, mask):
    """
    task : str
        Kind of ISC, must be in ['spatial', 'temporal']
    data_dir : str
        The path to the postprocess data directory on disk.
        Should contain all generated ISC maps.
    mask_img : str
        Path to the mask image on disk.
    """
    masker = NiftiMasker(mask_img=mask)
    
    path = Path(data_dir)
    files = path.rglob('ISC_*.nii.gz')
    subjects = sorted([str(f) for f in files], key=lambda x: int(
        re.search(f'sub-(\d+)', x).group(1)))

    # since ISC was fit to average of all other subjects,
    # only one set of values for each subject-task pairing
    isc = [masker.fit_transform(s) for s in subjects]
    corr = np.corrcoef(np.row_stack(isc))

    # Plot boundaries as boxes on top of timepoint correlation matrix
    _, ax = plt.subplots(1,1, figsize = (10,10))
    title_text = '''
    Correlation of individual Inter-Subject Correlation (ISC)
    maps for each presented movie.
    '''
    ax.imshow(corr, cmap = 'viridis')
    ax.set_title(title_text)
    ax.set_xlabel('Participants')
    ax.set_xticks(np.arange(0,36))
    ax.set_xticklabels(np.tile([
        'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06'],
        reps=6))
    ax.set_ylabel('Presented movies')
    ax.set_yticks(np.arange(0,36))
    ax.set_yticklabels(np.tile([
        'bourne', 'figures_run-1', 'life_run-1',
        'figures_run-2', 'life_run-2', 'wolf'],
        reps=6))
    
    # plot the boundaries 
    bounds = np.arange(-1, 37, 6, dtype='float') + 0.5
    bounds[0] += 0.1
    bounds[-1] -= 0.1
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                       edge, edge, fill=False, linewidth=2,
                                       edgecolor='w'))
    plt.show()