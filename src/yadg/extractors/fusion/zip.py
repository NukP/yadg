"""
For processing Inficon Fusion zipped data. This is a wrapper parser which unzips the
provided zip file, and then uses the :mod:`yadg.extractors.fusion.json` extractor
to parse every fusion-data file present in the archive.

Contains both the data from the raw chromatogram and the post-processed results.

Usage
`````
Available since ``yadg-4.0``.

.. autopydantic_model:: dgbowl_schemas.yadg.dataschema_5_1.filetype.Fusion_zip

Schema
``````
.. code-block:: yaml

    datatree.DataTree:
      coords:
        uts:              !!float
        species:          !!str
      data_vars:
        height:           (uts, species)        # Peak height at maximum
        area:             (uts, species)        # Integrated peak area
        concentration:    (uts, species)        # Calibrated peak area
        xout:             (uts, species)        # Mole fraction (normalized conc.)
        retention time:   (uts, species)        # Peak retention time
      {{ detector_name }}:
        coords:
          uts:            !!float               # Unix timestamp
          elution_time:   !!float               # Elution time
        data_vars:
          signal:         (uts, elution_time)   # Signal data
          valve:          (uts)                 # Valve position

Metadata
````````
No metadata is currently extracted.

.. codeauthor::
    Peter Kraus

"""

import zipfile
import tempfile
import os
from datatree import DataTree
from functools import reduce
import xarray as xr

from yadg.extractors.fusion.json import extract as extract_json
from yadg import dgutils


def extract(
    *,
    fn: str,
    timezone: str,
    encoding: str,
    **kwargs: dict,
) -> DataTree:
    zf = zipfile.ZipFile(fn)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        dt_list = []
        filenames = [ffn for ffn in os.listdir(tempdir) if ffn.endswith("fusion-data")]
        for ffn in sorted(filenames):
            path = os.path.join(tempdir, ffn)
            fdt = extract_json(fn=path, timezone=timezone, encoding=encoding, **kwargs)
            dt_list.append(fdt)
        # Merge the DataTree objects directly
        dt = reduce(merge_datatrees, dt_list)
        return dt
    
def merge_datatrees(dt1: DataTree, dt2: DataTree) -> DataTree:
    """
    Recursively merges two DataTree objects, ensuring all nodes and data are preserved.
    """
    # Merge datasets at the current node
    if dt1.ds is not None and dt2.ds is not None:
        try:
            dt1_ds = dt1.ds
            dt2_ds = dt2.ds
            dt1.ds = xr.concat(
                [dt1_ds, dt2_ds],
                dim="uts",
                combine_attrs="identical",
                coords="minimal",
                compat="no_conflicts",
            )
        except xr.MergeError as e:
            raise RuntimeError(
                f"Merging datasets at node '{dt1.name}' failed due to conflicting data variables or coordinates."
            ) from e
    elif dt1.ds is None:
        dt1.ds = dt2.ds
    # Merge attributes
    dt1.attrs.update(dt2.attrs)
    # Recursively merge child nodes
    for key in dt2.children:
        if key in dt1.children:
            dt1[key] = merge_datatrees(dt1[key], dt2[key])
        else:
            dt1[key] = dt2[key]
    return dt1


