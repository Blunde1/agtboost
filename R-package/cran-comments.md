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

## rhub::check_for_cran check results

on Fedora Linux, R-devel, clang, gfortran
0 errors | 0 warnings | 1 notes

* checking installed package size ... NOTE
  installed size is  8.0Mb
  sub-directories of 1Mb or more:
    libs   7.4Mb
    
This is due to not re-including debugging information in Makevars, as specified by the CRAN policy and pointed out by Prof Brian Ripley. A shared-object (.so) is larger than perhaps necessary, but compliant with CRAN policies.

## Changes in agtboost 0.9.3
Following e-mail from Prof Brian Ripley:
"The CRAN policy contains
- Packages should not attempt to disable compiler diagnostics, nor to
remove other diagnostic information such as symbols in shared objects."
Thus, agtboost 0.9.3 re-includes debugging information. This leads to package being larger on some OS.

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
