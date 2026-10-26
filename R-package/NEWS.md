------------------------------------------------------------------------
agtboost development version
------------------------------------------------------------------------

- Fix observation weights. Gradients and Hessians were multiplied by the weight vector with a matrix product (`g * w`) instead of element-wise (`g.cwiseProduct(w)`); with Eigen's runtime dimension checks disabled this scaled every observation by the first weight only, so weighted fits were identical to unweighted fits. Weights now enter gradients, Hessians, the initial prediction and the training loss for every loss function.
- Fix the reported training loss with weights (`y*w` was used in place of weighting each observation's loss) and the negative binomial training loss (`y*dispersion` in place of `y + dispersion`), which triggered spurious "Loss-function deviating from gradient boosting approximation" warnings.
- Weights are rescaled to mean one. The fit does not depend on the scale of the weights, but the training loss did, and with it the fixed-tolerance non-linearity check; the training loss is now the weighted mean loss.
- `gbt.train()` now errors on weights of the wrong length, negative, non-finite or all-zero weights, and weights supplied with `loss_function = "count::auto"` (which does not use them).
- Add tests for observation weights.

------------------------------------------------------------------------
agtboost 0.9.2 (2021-11-12)
------------------------------------------------------------------------

- Patch fixing slow gbt.save() functionality.
- Obtain XGBoost and LightGBM hyperparameters from gbt.complexity().
- Include attribute "offset" in gbt.train() and predict().
- Throw error when gb-loss-approximation deviates from true loss. Suggest lower learning_rate.
- Solves $\arg\min_\eta \sum_i l(y_i, g^{-1}(offset_i+\eta))$ numerically instead of simple average to obtain initial prediction
- Deprecate warnings for zero-inflation
- Decrease package size drastically by stripping debugging information 

------------------------------------------------------------------------
agtboost 0.9.1 (2020-10-04)
------------------------------------------------------------------------

- Patch fixing error from log(<int>) on Solaris. Fixed by casting problematic type to double.
- Fixing error on UBSAN: Undefined behaviour (method call from NULL pointer)
- Checked with rhub::check_with_sanitizers() and rhub::check_on_solaris()

------------------------------------------------------------------------
agtboost 0.9.0 (2020-08-12)
------------------------------------------------------------------------

- Initial release.
