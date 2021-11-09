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
