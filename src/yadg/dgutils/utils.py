import logging
import os
from typing import Union
from dgbowl_schemas.yadg import to_dataschema

from .. import dgutils

logger = logging.getLogger(__name__)


def calib_3to4(oldcal: dict, caltype: str) -> dict:
    newcal = {}
    if caltype == "calfile":
        for k, v in oldcal["detectors"].items():
            pd = {
                "window": (v.get("window", 3) - 1) // 2,
                "polyorder": v.get("poly", 2),
                "prominence": v.get("prominence", 1.0),
                "threshold": v.get("threshold", 1.0),
            }
            sp = {}
            for kk, vv in v["species"].items():
                if kk == "units" and vv == "min":
                    continue
                spec = {
                    "l": vv["l"] * 60.0,
                    "r": vv["r"] * 60.0,
                    "calib": {"inverse": {"slope": vv.get("rf", 1.0)}},
                }
                sp[kk] = spec
            id = {"det_1": 0, "det_2": 1}[v["id"]]
            newcal[k] = {"id": id, "peakdetect": pd, "species": sp}
    elif caltype == "Tcalfile":
        newcal = {"T": {"T_f": {"calib": {"linear": oldcal}}, "unit": "degC"}}
    elif caltype == "MFCcalfile":
        newcal = {}
        for k, v in oldcal.items():
            items = v.get("content", {k: 1.0})
            for kk, vv in items.items():
                if kk not in newcal:
                    newcal[kk] = {"unit": "ml/min"}
                newcal[kk][k] = {
                    "calib": {
                        "linear": {
                            "slope": v.get("slope", 1.0),
                            "intercept": v.get("intercept", 0.0),
                        }
                    },
                    "fraction": vv,
                }
    return newcal


def schema_3to4(oldschema: list) -> dict:
    newschema = {
        "metadata": {
            "provenance": {
                "type": "yadg update",
                "metadata": {
                    "yadg": dgutils.get_yadg_metadata(),
                    "update_schema": {"updater": "schema_3to4"},
                },
            },
            "version": "4.1",
            "timezone": "localtime",
        },
        "steps": [],
    }
    for oldstep in oldschema:
        newstep = {}

        newstep["parser"] = oldstep["datagram"]
        if newstep["parser"] == "gctrace":
            newstep["parser"] = "chromtrace"

        if "paths" in oldstep["import"]:
            oldstep["import"]["files"] = oldstep["import"].pop("paths")
        newstep["input"] = oldstep["import"]

        if oldstep.get("export", None) is not None:
            newstep["tag"] = oldstep["export"]

        parameters = {}
        for k, v in oldstep["parameters"].items():
            if k in ["Tcalfile", "MFCcalfile", "calfile"]:
                logger.warning(
                    "Parsing of post-processing parameter '%s' has been removed in "
                    "yadg-5.0, please use dgpost-2.0 to reproduce this functionality.",
                    k,
                )
            elif k == "method" and v == "q0refl":
                logger.warning(
                    "Parsing of post-processing parameter '%s' has been removed in "
                    "yadg-5.0, please use dgpost-2.0 to reproduce this functionality.",
                    k,
                )
            else:
                parameters[k] = v
        if parameters != {}:
            newstep["parameters"] = parameters
        newschema["steps"].append(newstep)

    return newschema


def update_schema(object: Union[list, dict]) -> dict:
    """
    Yadg's update worker function.

    This is the main function called when **yadg** is executed as ``yadg update``.
    The main idea is to allow a simple update pathway from older versions of `schema` and
    ``datagram`` files to the current latest and greatest.

    Currently supports:

     - updating ``DataSchema`` version 3.1 to 4.0 using routines in ``yadg``
     - updating ``DataSchema`` version 4.0 and above to the latest ``DataSchema``

    Parameters
    ----------
    object
        The object to be updated

    Returns
    -------
    newobj: dict
        The updated and validated `"datagram"` or `"schema"`.

    """

    if isinstance(object, list):
        logger.info("Updating list-style DataSchema")
        newobj = schema_3to4(object)
    elif isinstance(object, dict):
        logger.info("Updating dict-style DataSchema")
        newobj = object
    else:
        raise ValueError(f"Supplied object is of incorrect type: {type(object)}")
    newobj = to_dataschema(**newobj)
    while hasattr(newobj, "update"):
        newobj = newobj.update()
    return newobj


def schema_from_preset(preset: dict, folder: str) -> dict:
    if isinstance(preset["metadata"]["provenance"], str):
        preset["metadata"]["provenance"] = "yadg preset"
    elif isinstance(preset["metadata"]["provenance"], dict):
        preset["metadata"]["provenance"] = {
            "type": "yadg preset",
            "metadata": {"preset_provenance": preset["metadata"]["provenance"]},
        }
    for step in preset["steps"]:
        inpk = "import" if "import" in step else "input"
        filk = "files" if "files" in step[inpk] else "folders"
        newf = []
        for oldf in step[inpk][filk]:
            if os.path.isabs(oldf):
                logger.warning(
                    "Item '%s' in '%s' is an absolute path and will not be patched.",
                    oldf,
                    filk,
                )
            else:
                assert not oldf.startswith("." + os.path.sep), (
                    f"Item '{oldf}' in '{filk}' does start with '.{os.path.sep}' and "
                    f"therefore should not be patched using '{folder}'."
                )
                newp = os.path.abspath(os.path.join(folder, oldf))
                newf.append(newp)
        step[inpk][filk] = newf
        if "parameters" in step and "calfile" in step["parameters"]:
            oldf = step["parameters"]["calfile"]
            if os.path.isabs(oldf):
                logger.warning(
                    "Specified calfile '%s' is an absolute path "
                    "and will not be patched.",
                    oldf,
                )
            else:
                newp = os.path.abspath(os.path.join(folder, oldf))
                step["parameters"]["calfile"] = newp
        if "externaldate" in step:
            using = "from" if "from" in step["externaldate"] else "using"
            if "file" in step["externaldate"][using]:
                oldf = step["externaldate"][using]["file"]["path"]
                if os.path.isabs(oldf):
                    logger.warning(
                        "Specified externaldate file '%s' is an absolute path "
                        "and will not be patched.",
                        oldf,
                    )
                else:
                    newp = os.path.abspath(os.path.join(folder, oldf))
                    step["externaldate"][using]["file"]["path"] = newp
    return preset
