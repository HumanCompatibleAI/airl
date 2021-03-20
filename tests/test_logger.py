import csv
import os.path as osp
from collections import defaultdict

import stable_baselines3.common.logger as sb_logger

import imitation.util.logger as logger


def _csv_to_dict(csv_path: str) -> dict:
    result = defaultdict(list)
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v != "":
                    v = float(v)
                result[k].append(v)
    return result


def _compare_csv_lines(csv_path: str, expect: dict):
    observed = _csv_to_dict(csv_path)
    assert expect == observed


def test_no_accum(tmpdir):
    logger.configure(tmpdir, ["csv"])
    sb_logger.record("A", -1)
    sb_logger.record("A", 1)
    sb_logger.record("B", 1)
    sb_logger.dump()
    sb_logger.record("A", 2)
    sb_logger.dump()
    sb_logger.record("B", 3)
    sb_logger.dump()
    expect = {"A": [1, 2, ""], "B": [1, "", 3]}
    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect)


def test_hard(tmpdir):
    logger.configure(tmpdir)

    # Part One: Test logging outside of the accumulating scope, and within scopes
    # with two different different logging keys (including a repeat).

    sb_logger.record("no_context", 1)

    with logger.accumulate_means("disc"):
        sb_logger.record("C", 2)
        sb_logger.record("D", 2)
        sb_logger.dump()
        sb_logger.record("C", 4)
        sb_logger.dump()

    with logger.accumulate_means("gen"):
        sb_logger.record("E", 2)
        sb_logger.dump()
        sb_logger.record("E", 0)
        sb_logger.dump()

    with logger.accumulate_means("disc"):
        sb_logger.record("C", 3)
        sb_logger.dump()

    sb_logger.dump()  # Writes 1 mean each from "gen" and "disc".

    expect_raw_gen = {"raw/gen/E": [2, 0]}
    expect_raw_disc = {
        "raw/disc/C": [2, 4, 3],
        "raw/disc/D": [2, "", ""],
    }
    expect_default = {
        "mean/gen/E": [1],
        "mean/disc/C": [3],
        "mean/disc/D": [2],
        "no_context": [1],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(osp.join(tmpdir, "raw", "gen", "progress.csv"), expect_raw_gen)
    _compare_csv_lines(osp.join(tmpdir, "raw", "disc", "progress.csv"), expect_raw_disc)

    # Part Two:
    # Check that we append to the same logs after the first dump to "means/*".

    with logger.accumulate_means("disc"):
        sb_logger.record("D", 100)
        sb_logger.dump()

    sb_logger.record("no_context", 2)

    sb_logger.dump()  # Writes 1 mean from "disc". "gen" is blank.

    expect_raw_gen = {"raw/gen/E": [2, 0]}
    expect_raw_disc = {
        "raw/disc/C": [2, 4, 3, ""],
        "raw/disc/D": [2, "", "", 100],
    }
    expect_default = {
        "mean/gen/E": [1, ""],
        "mean/disc/C": [3, ""],
        "mean/disc/D": [2, 100],
        "no_context": [1, 2],
    }

    _compare_csv_lines(osp.join(tmpdir, "progress.csv"), expect_default)
    _compare_csv_lines(osp.join(tmpdir, "raw", "gen", "progress.csv"), expect_raw_gen)
    _compare_csv_lines(osp.join(tmpdir, "raw", "disc", "progress.csv"), expect_raw_disc)
