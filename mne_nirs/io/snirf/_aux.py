# Author: Robert Luke <code@robertluke.net>
#
# License: BSD (3-clause)

import logging

import h5py
import numpy as np
from mne.io import Raw
from pandas import DataFrame
from scipy import interpolate


class SnirfAux:
    """Container representing one SNIRF /nirs/aux(i) group."""

    def __init__(self, name, data, time, data_unit=None, time_offset=None):
        self.name = name
        self.data = data                # shape (n_samples, n_channels)
        self.time = time                # shape (n_samples,)
        self.data_unit = data_unit
        self.time_offset = time_offset


def read_snirf_aux(fname):
    """Read raw auxiliary data from SNIRF file.

    Parameters
    ----------
    fname : str
        Path to the SNIRF file.

    Returns
    -------
    aux_data : list of dict
        Each entry corresponds to one /nirs/aux(i) group:
        dict with keys:
            - name : str
            - dataTimeSeries : ndarray (n_samples, n_channels)
            - time : ndarray (n_samples,)
            - dataUnit : str (optional)
            - timeOffset : float (optional)
    """

    aux_out = []

    with h5py.File(fname, "r") as dat:

        basename = _get_nirs_basename(dat)
        aux_keys, aux_names = _get_aux_entries(dat)

        for key in sorted(aux_keys):
            g = dat[f"{base}/{key}"]

            entry = {}

            # Required in SNIRF spec
            entry["name"] = _decode_name(g["name"])
            entry["dataTimeSeries"] = np.array(g["dataTimeSeries"])
            entry["time"] = np.array(g["time"])

            # Optional in SNIRF spec
            if "dataUnit" in g:
                entry["dataUnit"] = _decode_name(g["dataUnit"])
            if "timeOffset" in g:
                entry["timeOffset"] = np.array(g["timeOffset"]).item()

            aux_out.append(entry)

    return aux_out


def read_snirf_aux_data(fname: str, raw: Raw):
    """Read auxiliary data from SNIRF file.

    Reads the auxiliary channel data (e.g. heart rate data,
    accelerometer, etc). The auxiliary data will be resampled
    to match the raw data.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    raw : Raw
        Instance of raw snirf data as created by read_raw_snirf.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe with aux info, resampled to NIRS data.
    """

    with h5py.File(fname, "r") as dat:

        basename = _get_nirs_basename(dat)
        aux_keys, aux_names = _get_aux_entries(dat)

        d = {"times": raw.times}
        for idx, aux in enumerate(aux_keys):
            aux_data = np.array(dat.get(f"{basename}/{aux}/dataTimeSeries"))
            aux_time = np.array(dat.get(f"{basename}/{aux}/time"))
            aux_data_interp = interpolate.interp1d(
                aux_time, aux_data, axis=0, bounds_error=False, fill_value="extrapolate"
            )
            aux_data_matched_to_raw = aux_data_interp(raw.times).flatten()
            d[aux_names[idx]] = aux_data_matched_to_raw

        df = DataFrame(data=d)
        df = df.set_index("times")

    return df


def _get_nirs_basename(dat: h5py.File):

    if "nirs" in dat:
        basename = "nirs"
    elif "nirs1" in dat:
        basename = "nirs1"
    else:
        raise RuntimeError("Data does not contain nirs field")
    return basename


def _get_aux_entries(dat: h5py.File):

    basename = _get_nirs_basename(dat)
    all_keys = list(dat.get(basename).keys())
    aux_keys = [i for i in all_keys if i.startswith("aux")]
    aux_names = [_decode_name(dat.get(f"{basename}/{k}/name")) for k in aux_keys]
    logging.debug(f"Found auxiliary channels {aux_names}")
    return aux_keys, aux_names


def _decode_name(key):

    if ds is None:
        return None

    # Prefer h5py’s string-safe accessor for h5py ≥ 3
    # otherwise fall back to legacy version/files
    try:
        return ds.asstr()[()]
    except Exception:
        return np.array(ds).item().decode()
