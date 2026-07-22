Reframed the workflows guide's hyperparameter-search section: mushin *does*
grid and random search (often the whole job for a small discrete space, with the
labeled dataset + stats as a bonus); it does not do adaptive/Bayesian search over
large or continuous spaces, which is where a dedicated optimizer like Optuna is
complementary. Replaces the earlier "mushin is not a hyperparameter optimizer"
overstatement.
