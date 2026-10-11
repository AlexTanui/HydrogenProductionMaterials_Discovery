# Current status: 200-epoch experiment finalized

See [completed experiment](final_200/Training_Budget_Report.md) and [analysis notebook](final_200/Training_Budget_Analysis.ipynb). The main Performance_Review.html now shows final 200-epoch validation results. Earlier 144-epoch snapshots are historical. No new test evaluation was performed.

---

# DimeNet++ performance analysis

## New work

This week I extended the existing DimeNet++ baseline with a reproducible performance analysis. I visualized training and validation, compared the saved experiments at their selected checkpoints, and examined held-out prediction errors. The test energy MAE is 1.607 kcal/mol and the force component MAE is 1.530 kcal/mol/angstrom. The energy predictions show a positive mean error of 1.576 kcal/mol. The squared mean bias accounts for 82.4% of energy mean squared error, so systematic bias is an important issue to investigate. I also broke down force errors by element and identified the ten configurations with the largest force errors. These are new diagnostics of the existing model, not results from a new training run. The next controlled experiment should use training and validation data to investigate energy calibration and loss weighting, with the current test results retained as the baseline.

## Interpretation and limits

- The best saved checkpoint occurs at epoch 100, the training budget limit; convergence is not established.
- Training metrics aggregate predictions while parameters are changing within each epoch; validation metrics use the end-of-epoch model.
- Force component MAE is different from force-vector norm error.
- Element errors: {"C": 2.356776624459773, "O": 1.9252432782836257, "H": 1.187957001440434} kcal/mol/angstrom. Differences do not establish a chemical cause.
- This is one molecule and one seed. Components and atoms within configurations are dependent; the plots are descriptive, with no independence-based confidence intervals.
- No test-derived offset was applied. The bias decomposition is a diagnostic identity, not a corrected model score.
- Test diagnostics are exploratory. Subsequent hyperparameters should be chosen on validation data; repeatedly consulted test data cannot provide a fresh unbiased final assessment.

## Proposed next experiment (not run)

1. Measure signed energy residuals on training and validation data for the frozen model.
2. Investigate energy loss weighting and a longer training budget, changing one factor at a time and selecting on validation only.
3. Repeat the chosen protocol with multiple seeds and report variation.
4. Agree with the team on a fresh final evaluation protocol before claiming further generalization gains.

## Reproduction

Run build_performance_analysis.py with --repo REPOSITORY --saved DIRECTORY_WITH_SAVED_PREDICTIONS --out OUTPUT_DIRECTORY. Requires NumPy and Matplotlib. The saved predictions are a local artifact and are not currently committed to GitHub. Gold data is read only.
