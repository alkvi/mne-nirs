# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import numpy as np
from scipy.linalg import eigh
from scipy.stats import zscore as _zscore
from sklearn.decomposition import PCA


def make_first_level_design_matrix(
    raw,
    stim_dur=1.0,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=(0,),
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
    oversampling=50,
):
    """
    Generate a design matrix based on annotations and model HRF.

    This is a wrapper function for the nilearn :footcite:`abraham2014machine`
    function ``make_first_level_design_matrix``. For detailed description
    of the arguments see the nilearn documentation at http://nilearn.github.io

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    stim_dur : Number
        The length of your stimulus.

    hrf_model : {'glover', 'spm', 'spm + derivative', \
        'spm + derivative + dispersion',\
        'glover + derivative', 'glover + derivative + dispersion',\
        'fir', None}, optional
        Specifies the hemodynamic response function. Default='glover'.

    drift_model : {'cosine', 'polynomial', None}, optional
        Specifies the desired drift model. Default='cosine'.

    high_pass : float, optional
        High-pass frequency in case of a cosine model (in Hz).
        Default=0.01.

    drift_order : int, optional
        Order of the drift model (in case it is polynomial).
        Default=1.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model (in scans). Default=[0].

    add_regs : array of shape(n_frames, n_add_reg) or pandas DataFrame
        Additional user-supplied regressors, e.g. data driven noise regressors
        or seed based regressors.

    add_reg_names : list of (n_add_reg,) str, optional
        If None, while add_regs was provided, these will be termed
        'reg_%i', i = 0..n_add_reg - 1
        If add_regs is a DataFrame, the corresponding column names are used
        and add_reg_names is ignored.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    oversampling : int, optional
        Oversampling factor used in temporal convolutions. Default=50.

    Returns
    -------
    design_matrix : DataFrame instance,
        Holding the computed design matrix, the index being the frames_times
        and each column a regressor.

    References
    ----------
    .. footbibliography::
    """
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = raw.times

    # Create events for nilearn
    conditions = raw.annotations.description
    onsets = raw.annotations.onset - raw.first_time
    duration = stim_dur * np.ones(len(conditions))
    events = DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    dm = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=drift_model,
        drift_order=drift_order,
        hrf_model=hrf_model,
        min_onset=min_onset,
        high_pass=high_pass,
        add_regs=add_regs,
        oversampling=oversampling,
        add_reg_names=add_reg_names,
        fir_delays=fir_delays,
    )

    return dm


def create_boxcar(raw, event_id=None, stim_dur=1):
    """
    Generate boxcar representation of the experimental paradigm.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    event_id : as specified in MNE
        Information about events.
    stim_dur : Number
        The length of your stimulus.

    Returns
    -------
    s : array
        Returns an array for each annotation label.
    """
    bc = np.ones(int(round(raw.info["sfreq"] * stim_dur)))
    events, ids = mne.events_from_annotations(raw, event_id=event_id)
    s = np.zeros((len(raw.times), len(ids)))
    for idx, _ in enumerate(ids):
        id_idx = [e[2] == idx + 1 for e in events]
        id_evt = events[id_idx]
        event_samples = [e[0] for e in id_evt]
        s[event_samples, idx] = 1.0
        s[:, idx] = np.convolve(s[:, idx], bc)[: len(raw.times)]
    return s


def longest_inter_annotation_interval(raw):
    """
    Compute longest ISI per annotation.

    Specifically, longest period between two trials of
    the same condition.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    longest : list
        Longest ISI per annotation.
    annotation_name : list
        Annotation name corresponding to reported interval.
    """
    annotation_name = np.unique(raw.annotations.description)
    longest = []
    for desc in annotation_name:
        mask = raw.annotations.description == desc
        longest.append(np.max(np.diff(raw.annotations.onset[mask])))
    return longest, annotation_name


def drift_high_pass(raw):
    """
    Compute cosine drift regressor high pass cut off.

    Value computed according to Nilearn :footcite:`abraham2014machine`
    `suggestion <http://nilearn.github.io/auto_examples/04_glm_first
    _level/plot_first_level_details.html#changing-the-drift-model>`__.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    cutoff : number
        Suggested high pass cut off.

    References
    ----------
    .. footbibliography::
    """
    longest, annotation_name = longest_inter_annotation_interval(raw)
    max_isi = np.max(longest)
    return 1 / (2 * max_isi)


def _temporal_embedding(aux, tau, n_emb):
    """
    Build a temporally embedded auxiliary matrix.

    Parameters
    ----------
    aux : array of shape (n_times, n_channels)
        Auxiliary signal matrix.
    tau : int
        Lag in samples between consecutive embeddings.
    n_emb : int
        Number of additional time-shifted copies to append.

    Returns
    -------
    emb : array of shape (n_times, n_channels * (n_emb + 1))
        Temporally embedded signal.
    """
    copies = [aux]
    for i in range(1, n_emb + 1):
        shifted = np.roll(aux, shift=i * tau, axis=0)
        shifted[: 2 * i] = shifted[2 * i]
        copies.append(shifted)
    return np.concatenate(copies, axis=1)


def _ledoit_wolf_cov(X):
    """
    Estimate covariance with Ledoit-Wolf optimal shrinkage.

    Parameters
    ----------
    X : array of shape (n_times, n_features)
        Data matrix (observations × features).

    Returns
    -------
    cov : array of shape (n_features, n_features)
        Regularised covariance estimate.
    shrinkage : float
        Optimal shrinkage coefficient selected by Ledoit-Wolf.
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(X)
    return lw.covariance_, lw.shrinkage_


def rtcca(X, aux, tau=1, n_emb=3, ct=0.3, pca_aux=False, pca_fnirs=False, shrink=True):
    r"""
    Extract GLM regressors using regularised temporally embedded CCA.

    Derives nuisance regressors from auxiliary signals (e.g. accelerometry)
    and fNIRS data using temporally embedded Canonical Correlation Analysis
    (tCCA) with optional Ledoit-Wolf covariance regularisation, as described
    in :footcite:`vonLuhmannEtAl2020`.

    Parameters
    ----------
    X : array of shape (n_times, n_fnirs_channels)
        fNIRS signal matrix.
    aux : array of shape (n_times, n_aux_channels)
        Auxiliary signal matrix (e.g. accelerometer). Feeding in
        PCA-orthogonalised data is advised.
    tau : int
        Lag in samples between consecutive temporal embeddings.
    n_emb : int
        Number of additional time-shifted copies to include in the
        temporally embedded auxiliary matrix.
    ct : float
        Canonical correlation threshold. Only components whose canonical
        correlation exceeds ``ct`` are returned as regressors.
    pca_aux : bool
        If ``True``, reduce ``aux`` to its principal components before
        temporal embedding.
    pca_fnirs : bool
        If ``True``, reduce ``X`` to its principal components before CCA.
    shrink : bool
        If ``True``, regularise auto-covariance matrices using Ledoit-Wolf
        optimal shrinkage :footcite:`LedoitWolf2004`.

    Returns
    -------
    regressors : array of shape (n_times, n_components)
        Noise regressors for use in a GLM design matrix. Contains only the
        CCA components whose canonical correlation exceeds ``ct``.
    info : dict
        Diagnostic information:

        ``'correlations'`` : array of shape (n_all_components,)
            Canonical correlations for all extracted components, sorted
            in descending order.
        ``'fnirs_sources'`` : array of shape (n_times, n_all_components)
            fNIRS data projected onto CCA filters.
        ``'aux_sources'`` : array of shape (n_times, n_all_components)
            Temporally embedded auxiliary data projected onto CCA filters.
        ``'fnirs_filters'`` : array
            CCA spatial filters for fNIRS.
        ``'aux_filters'`` : array
            CCA spatial filters for the temporally embedded auxiliary matrix.
        ``'aux_filters_reduced'`` : array
            Auxiliary filters retained after correlation thresholding.
        ``'aux_embedded'`` : array of shape (n_times, n_aux_channels * (n_emb + 1))
            Temporally embedded auxiliary matrix used in CCA.
        ``'shrinkage_fnirs'`` : float | None
            Ledoit-Wolf shrinkage coefficient for fNIRS (``None`` when
            ``shrink=False``).
        ``'shrinkage_aux'`` : float | None
            Ledoit-Wolf shrinkage coefficient for auxiliary signals (``None``
            when ``shrink=False``).

    Notes
    -----
    The algorithm follows :footcite:`vonLuhmannEtAl2020`:

    1. Optionally project ``aux`` and ``X`` onto their principal components.
    2. Construct a temporally embedded auxiliary matrix :math:`Y` by
       appending ``n_emb`` copies of ``aux``, each shifted by an integer
       multiple of ``tau`` samples.
    3. Z-score both :math:`X` and :math:`Y` (zero mean, unit variance per
       channel).
    4. Estimate auto-covariance matrices :math:`C_{XX}` and :math:`C_{YY}`
       with optional Ledoit-Wolf shrinkage, and the empirical cross-covariance
       :math:`C_{XY}`.
    5. Solve the symmetric generalised eigenvalue problem

       .. math::

           \begin{bmatrix} 0 & C_{XY} \\
                           C_{YX} & 0 \end{bmatrix} v = \lambda
           \begin{bmatrix} C_{XX} & 0 \\
                           0 & C_{YY} \end{bmatrix} v

       for canonical filters :math:`v` and canonical correlations
       :math:`\\lambda`.
    6. Project the temporally embedded auxiliary matrix through its CCA
       filter and retain components where
       :math:`\lambda > \rho_{\text{thresh}}` as noise regressors.

    References
    ----------
    .. footbibliography::
    """
    if pca_aux:
        aux = PCA().fit_transform(aux)
    if pca_fnirs:
        X = PCA().fit_transform(X)

    Y = _temporal_embedding(aux, tau, n_emb)

    n_times = min(X.shape[0], Y.shape[0])
    X = X[:n_times]
    Y = Y[:n_times]

    X = _zscore(X, axis=0)
    Y = _zscore(Y, axis=0)

    shrinkage_fnirs = None
    shrinkage_aux = None
    if shrink:
        Cxx, shrinkage_fnirs = _ledoit_wolf_cov(X)
        Cyy, shrinkage_aux = _ledoit_wolf_cov(Y)
    else:
        Cxx = X.T @ X / (n_times - 1)
        Cyy = Y.T @ Y / (n_times - 1)

    Cxy = X.T @ Y / (n_times - 1)
    Cyx = Cxy.T

    dx, dy = Cxy.shape
    A = np.block([[np.zeros((dx, dx)), Cxy], [Cyx, np.zeros((dy, dy))]])
    B = np.block([[Cxx, np.zeros((dx, dy))], [np.zeros((dy, dx)), Cyy]])

    eigvals, eigvecs = eigh(A, B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W_x = eigvecs[:dx]
    W_y = eigvecs[dx:]
    U = X @ W_x
    V = Y @ W_y

    mask = eigvals > ct
    regressors = V[:, mask]

    info = dict(
        correlations=eigvals,
        fnirs_sources=U,
        aux_sources=V,
        fnirs_filters=W_x,
        aux_filters=W_y,
        aux_filters_reduced=W_y[:, mask],
        aux_embedded=Y,
        shrinkage_fnirs=shrinkage_fnirs,
        shrinkage_aux=shrinkage_aux,
    )

    return regressors, info
