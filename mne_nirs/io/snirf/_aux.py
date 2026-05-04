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
    """Container representing one SNIRF /nirs/aux(i) group.

    Parameters
    ----------
    name : str
        Name of the auxiliary channel.
    data : ndarray, shape (n_samples, n_channels)
        The auxiliary time-series data.
    time : ndarray, shape (n_samples,)
        Time vector in TimeUnit units (see metaDataTags).
    data_unit : str or None
        SI unit of the auxiliary channel (optional).
    time_offset : float or None
        Offset of file time origin relative to absolute clock time (optional).
    """

    def __init__(self, name, data, time, data_unit=None, time_offset=None):
        self.name = name
        self.data = np.asarray(data)
        self.time = np.asarray(time)
        self.data_unit = data_unit
        self.time_offset = time_offset

    def __repr__(self):
        n_samples, n_ch = self.data.shape if self.data.ndim == 2 else (len(self.data), 1)
        return (
            f"<SnirfAux | name={self.name!r}, "
            f"n_samples={n_samples}, n_channels={n_ch}, "
            f"data_unit={self.data_unit!r}>"
        )


def read_snirf_aux(fname):
    """Read raw auxiliary data from a SNIRF file.

    Reads all /nirs/aux(i) groups and returns them as a list of
    :class:`SnirfAux` containers.

    Parameters
    ----------
    fname : str
        Path to the SNIRF file.

    Returns
    -------
    aux_list : list of SnirfAux
        One entry per /nirs/aux(i) group found in the file.
    """
    aux_out = []

    with h5py.File(fname, "r") as dat:
        basename = _get_nirs_basename(dat)
        aux_keys, _ = _get_aux_entries(dat)

        for key in sorted(aux_keys):
            g = dat[f"{basename}/{key}"]

            name = _decode_string(g["name"])
            data = np.array(g["dataTimeSeries"])
            time = np.array(g["time"])

            data_unit = _decode_string(g["dataUnit"]) if "dataUnit" in g else None
            time_offset = np.array(g["timeOffset"]).item() if "timeOffset" in g else None

            aux_out.append(SnirfAux(
                name=name,
                data=data,
                time=time,
                data_unit=data_unit,
                time_offset=time_offset,
            ))

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
    aux_names = [_decode_string(dat.get(f"{basename}/{k}/name")) for k in aux_keys]
    logging.debug(f"Found auxiliary channels {aux_names}")
    return aux_keys, aux_names


def _decode_string(ds):
    """Decode an h5py string dataset to a Python str."""
    if ds is None:
        return None

    # Prefer h5py's string-safe accessor (h5py >= 3),
    # otherwise fall back for legacy files.
    try:
        return ds.asstr()[()]
    except Exception:
        return np.array(ds).item().decode()
