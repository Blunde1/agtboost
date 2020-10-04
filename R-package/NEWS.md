------------------------------------------------------------------------
agtboost 0.9.0 (2020-08-12)
------------------------------------------------------------------------

- Initial release.

------------------------------------------------------------------------
agtboost 0.9.1 (2020-10-04)
------------------------------------------------------------------------

- Patch fixing error from log(<int>) on Solaris. Fixed by casting problematic type to double.
- Fixing error on UBSAN: Undefined behaviour (method call from NULL pointer)
- Checked with rhub::check_with_sanitizers() and rhub::check_on_solaris()

