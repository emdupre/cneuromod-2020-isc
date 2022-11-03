import numpy as np
import matplotlib as mpl
from pathlib import Path
from surfplot import Plot
import matplotlib.pyplot as plt
from surfplot.utils import threshold
from nilearn.maskers import NiftiMasker
from nilearn.image import threshold_img
from neuromaps.datasets import fetch_fsaverage
from neuromaps.transforms import mni152_to_fsaverage

u = 0.5
thr = 0.2
tasks = ['bourne', 'figures_run-1', 'life_run-1',
         'figures_run-2', 'life_run-2', 'wolf']
surfaces = fetch_fsaverage(density='164k')
lh, rh = surfaces['inflated']
sulc_lh, sulc_rh = surfaces['sulc']
path = Path.cwd()
mask_name = '../movie10.fmriprep/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
masker = NiftiMasker(mask_img=mask_name).fit()

plt.rcParams['svg.fonttype'] = 'none'

for t in tasks:
    isc_maps = path.glob(f'ISC_{t}_*')
    Z = [masker.transform(threshold_img(i, threshold=thr)) 
            for i in isc_maps]
    p = int((1 - u) * len(Z))
    Z_ = np.sort(np.vstack(Z).T, 1)

    # threshold and take conjunction
    conj = np.sum(Z_[:, :p], 1) / np.sqrt(p)
    conj_img = masker.inverse_transform(conj)
    conj_img.to_filename(f'conj_ISC_{t}.nii.gz')

    # average and then threshold
    avg = np.average(np.vstack(Z), 0)
    avg_img = threshold_img(masker.inverse_transform(avg), threshold=thr)
    avg_img.to_filename(f'conj_ISC_{t}.nii.gz')

    # project then threshold to avoid interpolation artefacts
    lh_gii, rh_gii = mni152_to_fsaverage(avg_img, fsavg_density='164k')
    lh_data = threshold(lh_gii.agg_data(), thr - 0.05)
    rh_data = threshold(rh_gii.agg_data(), thr - 0.05)

    p = Plot(lh, rh, size=(800, 200), zoom=1.2, layout='row', mirror_views=True)
    p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r', cbar=False)
    p.add_layer({'left': lh_data, 'right': rh_data}, cmap='YlOrRd', color_range=[0.2, 0.8])
    fig = p.build(colorbar=False)
    fig.savefig(f'avg_ISC_{t}.svg', transparent=True, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(1, 6))
fig.subplots_adjust(right=0.4)
cmap = mpl.cm.YlOrRd
norm = mpl.colors.Normalize(vmin=0.2, vmax=0.8)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='vertical',
             ticks=np.arange(0.2, 0.8, 0.2))
ax.tick_params(labelsize=20)
fig.savefig('colorbar.svg', transparent=True, bbox_inches='tight')

# Z = masker.transform(all_aligned).T
# p = int((1 - u) * len(Z.shape[1])
# Z_ = np.sort(Z, 1)
# conj = np.sum(Z_[:, :p], 1) / np.sqrt(p)
# path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
#     contrast, u, method, target_space, alignment_data))
# conj_img = masker.inverse_transform(conj)
# conj_img.to_filename(path)