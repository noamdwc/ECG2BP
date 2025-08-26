import os.path

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from .utils.constants import BASE_PATH
from .utils.hp_search_utils import objective

HP_SEARCH_DB = os.path.join(BASE_PATH, "hp_search", "search.db")
os.makedirs(os.path.dirname(HP_SEARCH_DB), exist_ok=True)


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True, n_startup_trials=20),
        pruner=HyperbandPruner(min_resource=3, max_resource=28, reduction_factor=3),
        study_name="ecg_hp_search",
        storage="sql:///" + HP_SEARCH_DB,  # so you can pause/resume
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=250, gc_after_trial=True)

    # Shrink the space (what actually matters?)
    from optuna.importance import get_param_importances
    print(get_param_importances(study, target=None))
    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)


if __name__ == '__main__':
    main()
