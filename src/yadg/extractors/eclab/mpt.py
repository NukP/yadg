"""
For processing of BioLogic's EC-Lab binary modular files.

Usage
`````
Available since ``yadg-4.0``.

.. autopydantic_model:: dgbowl_schemas.yadg.dataschema_5_1.filetype.EClab_mpt

Schema
``````
The ``.mpt`` files contain many columns that vary depending on the electrochemical
technique used. Below is shown a list of columns that can be expected to be present
in a typical ``.mpt`` file.

.. code-block:: yaml

    xarray.Dataset:
      coords:
        uts:            !!float     # Unix timestamp, without date
      data_vars:
        Ewe             (uts)       # Potential of the working electrode
        Ece             (uts)       # Potential of the counter electrode, if present
        I               (uts)       # Instantaneous current
        time            (uts)       # Time elapsed since the start of the experiment
        <Ewe>           (uts)       # Average Ewe potential since last data point
        <Ece>           (uts)       # Average Ece potential since last data point
        <I>             (uts)       # Average current since last data point
        ...

.. note::

     Note that in most cases, either the instantaneous or the averaged quantities are
     stored - only rarely are both available!

Notes on file structure
```````````````````````
These human-readable files are sectioned into headerlines and datalines.
The header part of the ``.mpt`` files is made up of information that can be found
in the settings, log and loop modules of the binary ``.mpr`` file.

If no header is present, the timestamps will instead be calculated from
the file's ``mtime()``.

Metadata
````````
The metadata will contain the information from the header of the file.

.. note ::

    The mapping between metadata parameters between ``.mpr`` and ``.mpt`` files
    is not yet complete.

.. codeauthor::
    Nicolas Vetsch

"""

import re
import logging
from babel.numbers import parse_decimal
from xarray import Dataset
from yadg import dgutils
from .common.techniques import get_resolution, technique_params, param_from_key
from .common.mpt_columns import column_units

logger = logging.getLogger(__name__)


def process_header(
    lines: list[str],
    timezone: str,
    locale: str,
) -> tuple[dict, list, dict]:
    """Processes the header lines.

    Parameters
    ----------
    lines
        The header lines, starting at line 3 (which is an empty line),
        right after the `"Nb header lines : "` line.

    Returns
    -------
    tuple[dict, dict]
        A dictionary containing the settings (and the technique
        parameters) and a dictionary containing the loop indexes.

    """
    sections = "\n".join(lines).split("\n\n")
    # Can happen that no settings are present but just a loops section.
    assert not sections[1].startswith("Number of loops : "), "no settings present"
    # Again, we need the acquisition time to get timestamped data.
    assert len(sections) >= 3, "no settings present"
    technique = sections[1].strip()
    settings_lines = sections[2].split("\n")
    technique, params_keys = technique_params(technique, settings_lines)
    params = settings_lines[-len(params_keys) :]

    # The sequence param columns are always allocated 20 characters.
    n_sequences = int(len(params[0]) / 20)
    params_values = []
    for seq in range(1, n_sequences):
        values = []
        for param in params:
            val = param[seq * 20 : (seq + 1) * 20]
            try:
                val = float(parse_decimal(val, locale=locale))
            except ValueError:
                val = val.strip()
            values.append(val)
        params_values.append(values)
    params = [dict(zip(params_keys, values)) for values in params_values]
    settings_lines = [line.strip() for line in settings_lines[: -len(params_keys)]]

    # Parse the acquisition timestamp.
    timestamp_re = re.compile(r"Acquisition started on : (?P<val>.+)")
    timestamp_match = timestamp_re.search("\n".join(settings_lines))
    timestamp = timestamp_match["val"]
    for format in ("%m/%d/%Y %H:%M:%S", "%m.%d.%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f"):
        uts = dgutils.str_to_uts(
            timestamp=timestamp, format=format, timezone=timezone, strict=False
        )
        if uts is not None:
            break
    if uts is None:
        raise NotImplementedError(f"Time format for {timestamp} not implemented.")

    loops = None
    if len(sections) >= 4 and sections[-1].startswith("Number of loops : "):
        # The header contains a loops section.
        loops_lines = sections[-1].split("\n")
        n_loops = int(loops_lines[0].split(":")[-1])
        indexes = []
        for n in range(n_loops):
            index = loops_lines[n + 1].split("to")[0].split()[-1]
            indexes.append(int(index))
        loops = {"n_loops": n_loops, "indexes": indexes}
    settings = {
        "posix_timestamp": uts,
        "technique": technique,
        "raw": "\n".join(lines),
    }
    return settings, params, loops


def process_data(
    lines: list[str],
    Eranges: list[float],
    Iranges: list[float],
    controls: list[str],
    locale: str,
):
    """Processes the data lines.

    Parameters
    ----------
    lines
        The data lines, starting right after the last header section.
        The first line is an empty line, the column names can be found
        on the second line.

    Returns
    -------
    dict
        A dictionary containing the datapoints in the format
        ([{column -> value}, ..., {column -> value}]). If the column
        unit is set to None, the value is an int. Otherwise, the value
        is a dict with value ("n"), sigma ("s"), and unit ("u").

    """
    # At this point the first two lines have already been read.
    # Remove extra column due to an extra tab in .mpt file column names.
    names = lines[1].split("\t")[:-1]
    units = dict()
    columns = list()
    for n in names:
        c, u = column_units[n]
        columns.append(c)
        if u is not None:
            units[c] = u
    data_lines = lines[2:]
    allvals = dict()
    allmeta = dict()
    for li, line in enumerate(data_lines):
        values = line.split("\t")
        vals = dict()
        for name, value in list(zip(columns, values)):
            if units.get(name) is None:
                ival = int(parse_decimal(value, locale=locale))
                if name == "I Range":
                    vals[name] = param_from_key("I_range", ival)
                else:
                    vals[name] = ival
            else:
                try:
                    fval = float(parse_decimal(value, locale=locale))
                    vals[name] = fval
                except ValueError:
                    sval = value.strip()
                    vals[name] = sval
        if "Ns" in vals:
            Erange = Eranges[vals["Ns"]]
            Irstr = Iranges[vals["Ns"]]
        else:
            Erange = Eranges[0]
            Irstr = Iranges[0]
        if "I Range" in vals:
            Irstr = vals["I Range"]
        Irange = param_from_key("I_range", Irstr, to_str=False)
        devs = {}
        if "control_V_I" in vals:
            icv = controls[vals["Ns"]]
            name = f"control_{icv}"
            vals[name] = vals.pop("control_V_I")
            units[name] = "mA" if icv in {"I", "C"} else "V"
        for col, val in vals.items():
            unit = units.get(col)
            if unit is None:
                continue
            assert isinstance(val, float), "`n` should not be string"
            devs[col] = get_resolution(col, val, unit, Erange, Irange)

        dgutils.append_dicts(vals, devs, allvals, allmeta, li=li)

    ds = dgutils.dicts_to_dataset(allvals, allmeta, units, fulldate=False)
    return ds


def extract(
    *,
    fn: str,
    encoding: str,
    locale: str,
    timezone: str,
    **kwargs: dict,
) -> Dataset:
    """Processes EC-Lab human-readable text export files.

    Parameters
    ----------
    fn
        The file containing the data to parse.

    encoding
        Encoding of ``fn``, by default "windows-1252".

    timezone
        A string description of the timezone. Default is "UTC".

    Returns
    -------
    :class:`xarray.Dataset`
        The full date may not be specified if header is not present.

    """
    file_magic = "EC-Lab ASCII FILE\n"
    with open(fn, "r", encoding=encoding) as mpt_file:
        assert mpt_file.read(len(file_magic)) == file_magic, "invalid file magic"
        mpt = mpt_file.read()
    lines = mpt.split("\n")
    nb_header_lines = int(lines[0].split()[-1])
    header_lines = lines[: nb_header_lines - 3]
    data_lines = lines[nb_header_lines - 3 :]
    settings, params = {}, []

    # Store current LC_NUMERIC before we do anything:
    if nb_header_lines <= 3:
        logger.warning("Header contains no settings and hence no timestamp.")
        start_time = 0.0
        fulldate = False
        Eranges = [20.0]
        Iranges = ["Auto"]
        ctrls = [None]
    else:
        settings, params, _ = process_header(header_lines, timezone, locale)
        start_time = settings.get("posix_timestamp")
        fulldate = True
        Eranges = []
        Iranges = []
        ctrls = []
        for el in params:
            E_range_max = el.get("E_range_max", float("inf"))
            E_range_min = el.get("E_range_min", float("-inf"))
            Eranges.append(E_range_max - E_range_min)
            Iranges.append(el.get("I_range", "Auto"))
            if "set_I/C" in el:
                ctrls.append(el["set_I/C"])
            elif "apply_I/C" in el:
                ctrls.append(el["apply_I/C"])
            else:
                ctrls.append(None)
    # Arrange all the data into the correct format.
    # TODO: Metadata could be handled in a nicer way.
    metadata = {"settings": settings, "params": params}

    ds = process_data(data_lines, Eranges, Iranges, ctrls, locale)
    if "time" in ds:
        ds["uts"] = ds["time"] + start_time
    else:
        ds["uts"] = [start_time]
    if fulldate:
        del ds.attrs["fulldate"]
    ds.attrs.update(metadata)
    # reset to original LC_NUMERIC
    return ds
