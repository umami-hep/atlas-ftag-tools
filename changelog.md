# Changelog

### [Latest]
- Cache the estimated available jets [#67](https://github.com/umami-hep/atlas-ftag-tools/pull/67)
- Introduce script to calculate efficiencies and rejections at a given discriminant cut value [#64](https://github.com/umami-hep/atlas-ftag-tools/pull/64)

### [v0.1.14]
- Add option to set custom flavour yaml path [#65](https://github.com/umami-hep/atlas-ftag-tools/pull/65)
- Extend working point script to Xbb [#55](https://github.com/umami-hep/atlas-ftag-tools/pull/55)

### [v0.1.13]
- Add common git check functions  [#62](https://github.com/umami-hep/atlas-ftag-tools/pull/62)
- Added 'X % M != n' cut [#61](https://github.com/umami-hep/atlas-ftag-tools/pull/61)

### [v0.1.12]
- Update top labelling names [#39](https://github.com/umami-hep/atlas-ftag-tools/pull/39)
- Add isolation classes [#58](https://github.com/umami-hep/atlas-ftag-tools/pull/58)

### [v0.1.11]
- Bugfix of `structured_from_dict` function [#53](https://github.com/umami-hep/atlas-ftag-tools/pull/54)

### [v0.1.10]
- Add func to create structured array from dict of arrays [#52](https://github.com/umami-hep/atlas-ftag-tools/pull/52)
- Copy attrs when splitting files [#51](https://github.com/umami-hep/atlas-ftag-tools/pull/51)
- Update linter versions

### [v0.1.9]
- Fix bug in estimate_available_jets [#50](https://github.com/umami-hep/atlas-ftag-tools/pull/50)

### [v0.1.8]
- Fix to h5split to allow files with remainders [#49](https://github.com/umami-hep/atlas-ftag-tools/pull/49)
- Backward compatibility to python3.8 and small fix for h5split.py [#48](https://github.com/umami-hep/atlas-ftag-tools/pull/48)
- Add h5split command [#47](https://github.com/umami-hep/atlas-ftag-tools/pull/47)

### [v0.1.7]
- Improvements to the H5Writer interface [#46](https://github.com/umami-hep/atlas-ftag-tools/pull/46)

### [v0.1.6]
- Add on the fly variable renaming and data transforms [#44](https://github.com/umami-hep/atlas-ftag-tools/pull/44)
- Revert [#43](https://github.com/umami-hep/atlas-ftag-tools/pull/43) as it breaks downstream code that relies on hashing `Flavour` objects
- Unfreeze flavour class to allow for on the fly modification [#43](https://github.com/umami-hep/atlas-ftag-tools/pull/43)
- Update linters [#41](https://github.com/umami-hep/atlas-ftag-tools/pull/41)

### [v0.1.5]
- Improve estimate_available_jets tests [#38](https://github.com/umami-hep/atlas-ftag-tools/pull/38)
- Improve estimate_available_jets for multi-sample case [#37](https://github.com/umami-hep/atlas-ftag-tools/pull/37)
- Add equal_jets option to for multi-sample H5Reader [#33](https://github.com/umami-hep/atlas-ftag-tools/pull/33)
- Changing labels of Hbb and Hcc [#35](https://github.com/umami-hep/atlas-ftag-tools/pull/35)
- Setting num_jets to maximum available if too large [#34](https://github.com/umami-hep/atlas-ftag-tools/pull/34)

### [v0.1.4]
- Check for `num_jets` of `-1` and add new inclusive top category [#31](https://github.com/umami-hep/atlas-ftag-tools/pull/31)
- Update test for vds.py [#30](https://github.com/umami-hep/atlas-ftag-tools/pull/30)
- Update cuts to allow larger possible max integer selection [#29](https://github.com/umami-hep/atlas-ftag-tools/pull/29)
- Add codes to copy attributes from source files to the target file [#20](https://github.com/umami-hep/atlas-ftag-tools/pull/20/)

### [v0.1.3]
- Fix for sample class [#28](https://github.com/umami-hep/atlas-ftag-tools/pull/28/)
- Add test for H5Reader

### [v0.1.2]
- Fix shuffling bug in H5Reader [#26](https://github.com/umami-hep/atlas-ftag-tools/pull/26)
- Update `working_points.py` with calculation of WPs given rejections [#23](https://github.com/umami-hep/atlas-ftag-tools/pull/23)

### [v0.1.1]
- Replace git hash with package version in output files [#25](https://github.com/umami-hep/atlas-ftag-tools/pull/25)
- Add tests [#24](https://github.com/umami-hep/atlas-ftag-tools/pull/24)

### [v0.1.0]
- Update `working_points.py` with configurable cuts [#18](https://github.com/umami-hep/atlas-ftag-tools/pull/18)
- Add `working_points.py` script to calculate working points [#17](https://github.com/umami-hep/atlas-ftag-tools/pull/17)

### [v0.0.8]
- Add `vds` command line tool to create virtual datasets [#16](https://github.com/umami-hep/atlas-ftag-tools/pull/16)
- fix for [#14](https://github.com/umami-hep/atlas-ftag-tools/pull/14)

### [v0.0.7]
- Improve Sample class [#15](https://github.com/umami-hep/atlas-ftag-tools/pull/15)
- make `inf` check optional [#14](https://github.com/umami-hep/atlas-ftag-tools/pull/14)

### [v0.0.6]
- minor `rng` fixes, add minimum package requirements [#13](https://github.com/umami-hep/atlas-ftag-tools/pull/13)

### [v0.0.5]
- support python >=3.8, unpin package versions [#12](https://github.com/umami-hep/atlas-ftag-tools/pull/12)
- improve mock data generation, update H5Reader defaults [#10](https://github.com/umami-hep/atlas-ftag-tools/pull/10)
- add git hash to files written with H5Writer [#9](https://github.com/umami-hep/atlas-ftag-tools/pull/9)

### [v0.0.4]
- set uniform python version in docker and actions [#7](https://github.com/umami-hep/atlas-ftag-tools/pull/7)
- add function to estimate number of available jets, support python 3.10 [#6](https://github.com/umami-hep/atlas-ftag-tools/pull/6)

### [v0.0.3]

first working release

### [v0.0.2]

fixed version of buggy release

### [v0.0.1]

initial buggy release
