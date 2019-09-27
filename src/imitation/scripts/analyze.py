from collections import OrderedDict
import os.path as osp
from typing import Optional

import pandas as pd
from sacred.observers import FileStorageObserver

from imitation.config.analyze import anal_ex
import imitation.util.sacred as sacred_util


@anal_ex.main
def analyze_imitation(source_dir: str,
                      run_name: Optional[str],
                      csv_output_path: Optional[str],
                      verbose: bool,
                      ) -> pd.DataFrame:
  """
  Args:
    source_dir: A directory containing Sacred FileObserver subdirectories
      associated with the `train_adversarial` Sacred script. Behavior is
      undefined if there are Sacred subdirectories associated with other
      scripts.
    run_name: If provided, then only analyze results from Sacred directories
      associated with this run name. `run_name` is compared against the
      "experiment.name" key in `run.json`.
    csv_output_path: If provided, then save a CSV output file to this path.
    verbose: If True, then print the dataframe.

  Returns:
    A list of dictionaries used to generate the analysis DataFrame.
  """
  sacred_dirs = sacred_util.filter_subdirs(source_dir)
  sacred_dicts = (sacred_util.SacredDicts.load_from_dir(sacred_dir)
                  for sacred_dir in sacred_dirs)
  if run_name is not None:
    sacred_dicts = filter(
      lambda sd: sacred_util.dict_get_nested(sd, "experiment.name") == run_name,
      sacred_dicts)

  rows = []
  for sd in sacred_dicts:
    row = OrderedDict()
    rows.append(row)

    imit_stats = sd.result["imit_stats"]
    expert_stats = sd.result["expert_stats"]
    row["use_gail"] = sd.config["init_trainer_kwargs"]["use_gail"]
    row["env_name"] = sd.config["env_name"]
    row["n_expert_demos"] = sd.config["n_expert_demos"]
    row["run_name"] = sd.run["experiment"]["name"]
    row["expert_return_summary"] = _make_reward_summary(expert_stats)
    row["imit_return_summary"] = _make_reward_summary(imit_stats, "monitor")
    row["imit_vs_expert_return"] = (expert_stats["reward_mean"] /
                                    imit_stats["monitor_reward_mean"])
    row["imit_return_mean"] = imit_stats["monitor_reward_mean"]
    row["imit_return_std_dev"] = imit_stats["monitor_reward_std"]

  df = pd.DataFrame(rows)
  df.to_csv(csv_output_path)

  if verbose:
    print(df)
  return rows


def _make_reward_summary(stats: dict, prefix="") -> str:
  return "{:3g} ± {:3g} (n={})".format(
    stats[f"{prefix}reward_mean"],
    stats[f"{prefix}reward_std"],
    stats["n_traj"])


def main_console():
  observer = FileStorageObserver.create(
    osp.join('output', 'sacred', 'analyze'))
  anal_ex.observers.append(observer)
  anal_ex.run_commandline()


if __name__ == "__main__":
  main_console()
