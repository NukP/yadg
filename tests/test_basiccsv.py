import pytest
from utils import (
    datagram_from_input,
    standard_datagram_test,
    pars_datagram_test,
    datadir,
)


@pytest.mark.parametrize(
    "input, ts",
    [
        (
            {  # ts1 - units on 2nd line, correct number of rows, correct value
                "case": "case_uts_units.csv"
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 6,
                "point": 0,
                "pars": {"flow": {"sigma": 0.0, "value": 15.0, "unit": "ml/min"}},
            },
        ),
        (
            {  # ts2 - timestamp from uts and index
                "case": "case_uts_units.csv",
                "parameters": {"timestamp": {"uts": {"index": 0}}},
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 6,
                "point": 2,
                "pars": {"uts": {"value": 1631626610.0}},
            },
        ),
        (
            {  # ts3 - semicolon separator, rtol for flow, but not for T
                "case": "case_timestamp.ssv",
                "parameters": {
                    "sep": ";",
                    "units": {"flow": "ml/min", "T": "K", "p": "atm"},
                    "sigma": {"flow": {"rtol": 0.1}},
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 7,
                "point": 0,
                "pars": {
                    "flow": {"sigma": 1.5, "value": 15.0, "unit": "ml/min"},
                    "T": {"sigma": 0.0, "value": 23.1, "unit": "K"},
                },
            },
        ),
        (
            {  # ts4 - semicolon separator, timestamp from timestamp and index
                "case": "case_timestamp.ssv",
                "parameters": {
                    "sep": ";",
                    "units": {"flow": "ml/min", "T": "K", "p": "atm"},
                    "timestamp": {"timestamp": {"index": 0}},
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 7,
                "point": 3,
                "pars": {"uts": {"value": 1631284405.0}},
            },
        ),
        (
            {  # ts5 - tab separator, sigma for T from atol
                "case": "case_custom_ts.tsv",
                "parameters": {
                    "sep": "\t",
                    "timestamp": {
                        "timestamp": {"index": 1, "format": "%d.%m.%Y %I:%M:%S%p"}
                    },
                    "sigma": {"T": {"atol": 0.05}},
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 5,
                "point": 3,
                "pars": {"T": {"value": 351.2, "sigma": 0.05, "unit": "K"}},
            },
        ),
        (
            {  # ts6 - tab separator, timestamp from timestamp, format, index
                "case": "case_custom_ts.tsv",
                "parameters": {
                    "sep": "\t",
                    "timestamp": {
                        "timestamp": {"index": 1, "format": "%d.%m.%Y %I:%M:%S%p"}
                    },
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 5,
                "point": 4,
                "pars": {"uts": {"value": 1631280585.0}},
            },
        ),
        (
            {  # ts7 - tab separator, timestamp from timestamp, format, index
                "case": "case_custom_ts.tsv",
                "parameters": {
                    "sep": "\t",
                    "timestamp": {
                        "timestamp": {"index": 1, "format": "%d.%m.%Y %I:%M:%S%p"}
                    },
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 5,
                "point": 2,
                "pars": {"uts": {"value": 1631276985.0}},
            },
        ),
        (
            {  # ts8 - tab separator, timestamp from timestamp, format, index
                "case": "case_custom_ts.tsv",
                "parameters": {
                    "sep": "\t",
                    "timestamp": {
                        "timestamp": {"index": 1, "format": "%d.%m.%Y %I:%M:%S%p"}
                    },
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 5,
                "point": 0,
                "pars": {"uts": {"value": 1631273385.0}},
            },
        ),
        (
            {  # ts9 - tab separator, timestamp from iso date and iso time
                "case": "case_date_time_iso.csv",
                "parameters": {
                    "timestamp": {"date": {"index": 0}, "time": {"index": 1}}
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 4,
                "point": 1,
                "pars": {"uts": {"value": 1610659800.0}},
            },
        ),
        (
            {  # ts10 - timestamp from time with custom format only
                "case": "case_time_custom.csv",
                "parameters": {
                    "timestamp": {"time": {"index": 0, "format": "%I.%M%p"}}
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 3,
                "point": 0,
                "pars": {"uts": {"value": 43140}},
            },
        ),
        (
            {  # ts11 - timestamp from time with custom format only
                "case": "case_time_custom.csv",
                "parameters": {
                    "timestamp": {"time": {"index": 0, "format": "%I.%M%p"}}
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 3,
                "point": 1,
                "pars": {"uts": {"value": 43200}},
            },
        ),
        (
            {  # ts12 - convert functionality test
                "case": "case_timestamp.ssv",
                "parameters": {
                    "sep": ";",
                    "units": {"flow": "ml/min", "T": "K", "p": "atm"},
                    "convert": {
                        "flow": {
                            "flow": {
                                "calib": {"linear": {"slope": 1e-6 / 60}, "atol": 1e-8}
                            },
                            "unit": "m3/s",
                        }
                    },
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 7,
                "point": 0,
                "pars": {
                    "flow": {
                        "sigma": 0.0,
                        "value": 15.0,
                        "unit": "ml/min",
                        "raw": True,
                    },
                    "flow": {
                        "sigma": 1e-8,
                        "value": 2.5e-7,
                        "unit": "m3/s",
                        "raw": False,
                    },
                },
            },
        ),
        (
            {  # ts13 - convert functionality with intercept only and both global and calib sigma
                "case": "case_uts_units.csv",
                "parameters": {
                    "atol": 0.1,
                    "convert": {
                        "T": {
                            "T": {
                                "calib": {"linear": {"intercept": 273.15}, "atol": 0.5}
                            },
                            "unit": "K",
                        }
                    },
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 6,
                "point": 0,
                "pars": {
                    "T": {"sigma": 0.1, "value": 23.1, "unit": "°C", "raw": True},
                    "T": {"sigma": 0.5, "value": 296.25, "unit": "K", "raw": False},
                },
            },
        ),
        (
            {  # ts14 - calfile functionality
                "case": "case_uts_units.csv",
                "parameters": {"atol": 0.1, "calfile": "calib.json"},
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 6,
                "point": 0,
                "pars": {
                    "T": {"sigma": 0.1, "value": 296.25, "unit": "K", "raw": False}
                },
            },
        ),
        (
            {  # ts15 - calfile functionality with fractions and total
                "case": "measurement.csv",
                "parameters": {
                    "sep": ";",
                    "timestamp": {
                        "timestamp": {"index": 0, "format": "%Y-%m-%d-%H-%M-%S"}
                    },
                    "calfile": "tfcal.json",
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 1662,
                "point": 0,
                "pars": {
                    "C3H8": {
                        "sigma": 0.0,
                        "value": 0.0,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "N2": {
                        "sigma": 0.0,
                        "value": 30.361,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "O2": {
                        "sigma": 0.0,
                        "value": 1.579,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "total": {
                        "sigma": 0.030,
                        "value": 31.940,
                        "unit": "ml/min",
                        "raw": False,
                    },
                },
            },
        ),
        (
            {  # ts16 - calfile functionality with fractions and total
                "case": "measurement.csv",
                "parameters": {
                    "sep": ";",
                    "timestamp": {
                        "timestamp": {"index": 0, "format": "%Y-%m-%d-%H-%M-%S"}
                    },
                    "calfile": "tfcal.json",
                },
            },
            {
                "nsteps": 1,
                "step": 0,
                "nrows": 1662,
                "point": 100,
                "pars": {
                    "C3H8": {
                        "sigma": 0.0,
                        "value": 1.204,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "N2": {
                        "sigma": 0.0,
                        "value": 35.146,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "O2": {
                        "sigma": 0.0,
                        "value": 3.577,
                        "unit": "ml/min",
                        "raw": False,
                    },
                    "total": {
                        "sigma": 0.035,
                        "value": 39.927,
                        "unit": "ml/min",
                        "raw": False,
                    },
                },
            },
        ),
    ],
)
def test_datagram_from_basiccsv(input, ts, datadir):
    ret = datagram_from_input(input, "basiccsv", datadir)
    standard_datagram_test(ret, ts)
    pars_datagram_test(ret, ts)
