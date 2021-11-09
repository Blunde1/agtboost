------------------------------------------------------------------------
agtboost 0.9.2 (2021-11-09)
------------------------------------------------------------------------
## Test environments
* local darwin17 install, R 4.1.1
* winbuilder
* devtools check
* rhub check_for_cran, check_on_solaris, check_with_sanitizers

## R CMD check results

0 errors | 0 warnings | 1 notes

* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Berent Ånund Strømnes Lunde <lundeberent@gmail.com>'

## Changes in agtboost 0.9.2
Some patching of serialization/deserialization of model objects, along with new features for model training. A deprecation notice is included, and package should be smaller on some OS due to stripping debugging information.
