import re
import warnings
import itertools
import nibabel as nib
from pathlib import Path
from nilearn import image, input_data
from nilearn.interfaces import fmriprep

subjects = ['sub-01', 'sub-02', 'sub-03',
            'sub-04', 'sub-05', 'sub-06']


def _get_segment(files, task):
    """
    Gets the segment of the movie (as encoded in the task
    value) for sorting. Since some segments were shown out
    of order, this is more reliable than a naive sort of
    the filenames.

    Parameters
    ----------
    files : list of str
        The list of filenames that should be sorted.

    task : str
        The name of the task, not encoding the segment.
        For example, if the task value is 'figures04'
        this string should be 'figures'.

    Returns
    -------
    segments: list of str
        The list of filenames, sorted by movie segment.
    """
    segments = sorted(files, key=lambda x: int(
        re.search(f'task-{task}(\d+)', x).group(1)))
    return segments


def _nifti_mask_movie(scan, mask, smoothing_fwhm=None):
    """
    Cleans movie data, including standardizing, detrending,
    and high-pass filtering at 0.01Hz. Corrects for supplied
    confounds. Optionally smooths time series.

    Parameters
    ----------
    scan: niimg_like
        An in-memory niimg
    mask: str
        The (brain) mask within which to process data.
    confounds: np.ndarray
        Any confounds to correct for in the cleaned data set.
    """
    # niftimask and clean data
    # high_pass=0.01, low_pass=0.1,
    masker = input_data.NiftiMasker(mask_img=mask, t_r=1.49,
                                    standardize=True, detrend=True,
                                    high_pass=0.01, low_pass=0.1,
                                    smoothing_fwhm=smoothing_fwhm)
    confounds, _ = fmriprep.load_confounds_strategy(
        scan, denoise_strategy="simple")
    cleaned = masker.fit_transform(scan, confounds=confounds)
    return masker.inverse_transform(cleaned)


def create_data_dictionary():
    """
    Creates a data_dictionary for easily accessing all of the relevant
    parameters of a movie10 task.

    Returns
    -------
    data_dictionary :  dict
        A dictionary with the following fields:
         - segment_lengths : list
            The number of frames to consider in each segment
         - regr_str : str
            The string to use when querying regressors
         - tmpl_str : str
            The string to use when querying BOLD files, containing
            the template name
    """
    # segment lengths varied slightly across subjects. `segment_length.py`
    # calculates the minimum segment length across subjects, indexed here.
    # this allows us to ensure that all subjects have the same amount of data.
    segment_lengths = [
        [403, 405, 405, 405, 405, 405, 405, 405, 405, 380],
        [406, 406, 406, 406, 406, 406, 406, 406, 406, 406,
         406, 406, 406, 406, 406, 406, 498],
        [402, 409, 410, 409, 408, 408, 408, 409, 409, 409, 409, 373],
        [402, 409, 410, 409, 408, 408, 408, 409, 409, 409, 409, 373],
        [406, 406, 406, 406, 384],
        [406, 406, 406, 406, 384]
    ]
    runs = ['', '', 'run-1_', 'run-2_', 'run-1_', 'run-2_']

    tmpl_str = 'space-MNI152NLin2009cAsym_desc-preproc_bold'

    to_process = []
    tasks = ['bourne', 'wolf', 'figures', 'figures', 'life', 'life']
    for t, run, seg_len in zip(tasks, runs, segment_lengths):
        task_dictionary = {
            'task' : t,
            'segment_lengths': seg_len,
            'run_num': run,
            'tmpl_str': tmpl_str}
        to_process.append(task_dictionary)

    return to_process


def subset_and_process_movie10(bold_files,
                               task_dict, subject,
                               n_segments=None, fwhm=None):
    """
    Subsets and preprocesses each movie segment to the minimum length
    available across all subjects. Please see the function 
    `create_data_dictionary` for all utilized lengths.

    Note that the movie10 tasks are long, with the longest movie
    (Wolf of Wall Street) clocking in at almost 7000 frames. This function is
    therefore also designed to subset the movie to a set number of segments;
    for example, to match the number of frames across tasks. Although this
    behavior is off by default (and controllable with the n_segments
    argument), note that if you choose to process the whole movie you can
    expect a very high memory usage.

    Parameters
    ----------
    bold_files : list
        A list of the BOLD file names to subset and process.
    segment_lengths : list
        A list of the number of frames to consider from each segment.
    task_dict : dict
    subject : str
    n_segments : int
        The number of segments to subset from the movie.
        Will error if the number of segments requested is more than are
        available in the movie.
    fwhm : int
        The size of the Gaussian smoothing kernel to apply.

    Returns
    -------
    postproc_fname : str
        Filename for the concatenated postprocessed file, correcting for
        the provided confounds and optionally smoothed to supplied FWHM.
    """
    # use the brain mask directly from templateflow, so all subjects have
    # the same number of voxels.
    tpl_mask = './tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
    task, segment_lengths, run, tmpl = task_dict.values()

    if n_segments is not None:
        # We could easily see user error here; force a check
        try:
            assert n_segments <= len(bold_files)
        except AssertionError:
            warnings.warn(
                f'Missing BOLD files for {subject} : {task} {run}')
            return
        movie_segments = bold_files[:n_segments]

    postproc_fname = f'{subject}/{subject}_task-{task}_{run}{tmpl}.nii.gz'.replace(
        'desc-preproc', f'desc-fwhm{fwhm}')
    if Path(postproc_fname).exists():
        print(f'File {postproc_fname} already exists; skipping')
        return postproc_fname

    postproc_segments = []
    for n, m in enumerate(movie_segments):
        postproc = _nifti_mask_movie(
            scan=m, mask=tpl_mask, smoothing_fwhm=fwhm)
        postproc_segments.append(postproc.slicer[..., :segment_lengths[n]])

    movie = image.concat_imgs(postproc_segments)
    nib.save(movie, postproc_fname)
    return postproc_fname


if __name__ == "__main__":

    task_dicts = create_data_dictionary()
    for s, t in itertools.product(subjects, task_dicts):

        task, seg_len, run, tmpl = t.values()

        print(f'*{task}*{run}{tmpl}.nii.gz')
        files = Path(s).rglob(f'*{task}*{run}{tmpl}.nii.gz')

        # get the sorted movie segments. check they're right w:
        # print(*segments, sep='\n')
        movie_segments = _get_segment([str(f) for f in files], task)
        print(*movie_segments, sep='\n')

        subset_and_process_movie10(movie_segments, t, s,
                                   n_segments=5, fwhm=6)

        # this is slow, so keep track of subject, task
        print(s, t)
