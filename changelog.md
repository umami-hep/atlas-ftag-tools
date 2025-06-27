# Changelog

### [Latest]

- Adding correct Usage of NaNs in Cuts [#134](https://github.com/umami-hep/atlas-ftag-tools/pull/134)

### [v0.2.13]

- Update Minimum Python Version to 3.10 [#131](https://github.com/umami-hep/atlas-ftag-tools/pull/131)
- Update Virtual Dataset Creation to Incorporate the Cut Bookkeeper [#132](https://github.com/umami-hep/atlas-ftag-tools/pull/132)

### [v0.2.12]

- Minor qcdnonbb definition update [#127](https://github.com/umami-hep/atlas-ftag-tools/pull/127)
- Ensure that the mock file generation is deterministic [#130](https://github.com/umami-hep/atlas-ftag-tools/pull/130)
- Remove flavours sorting in Labeller [#124](https://github.com/umami-hep/atlas-ftag-tools/pull/124)
- Allow writer to have varying number of jets, fix reader batch sizes [#125](https://github.com/umami-hep/atlas-ftag-tools/pull/125)

### [v0.2.11]

- Allow writer to write some things at full precision, and the rest at half [#123](https://github.com/umami-hep/atlas-ftag-tools/pull/123)
- Regex integration for vds merging [#116](https://github.com/umami-hep/atlas-ftag-tools/pull/116)
- Update xbb classes definitions [#122](https://github.com/umami-hep/atlas-ftag-tools/pull/122)
- Fixing Issues in the Imports & Update Docs [#120](https://github.com/umami-hep/atlas-ftag-tools/pull/120)
- Updating new Sphinx Docs [#119](https://github.com/umami-hep/atlas-ftag-tools/pull/119)
- Adding new Sphinx Docs [#118](https://github.com/umami-hep/atlas-ftag-tools/pull/118)

### [v0.2.10]

- Adding fraction optimization script [#117](https://github.com/umami-hep/atlas-ftag-tools/pull/117)
- Add flavours for trigger Xbb tagging based on delta R matching [#115](https://github.com/umami-hep/atlas-ftag-tools/pull/115)
- Moving big parts of the puma metrics into atlas-ftag-tools[#113](https://github.com/umami-hep/atlas-ftag-tools/pull/113)
- Remove docker completely [#114](https://github.com/umami-hep/atlas-ftag-tools/pull/114)

### [v0.2.9]

- Adding new workingpoint calculation [#111](https://github.com/umami-hep/atlas-ftag-tools/pull/111)
- Fixing tests [#109](https://github.com/umami-hep/atlas-ftag-tools/pull/109)
- Adding label for GN2XTau [#108](https://github.com/umami-hep/atlas-ftag-tools/pull/108)
- Splitting light-jet class definition [#107](https://github.com/umami-hep/atlas-ftag-tools/pull/107)

### [v0.2.8]

- Update isolation flavour labels [#106](https://github.com/umami-hep/atlas-ftag-tools/pull/106)

### [v0.2.7]
- Add weights to sample class [#104](https://github.com/umami-hep/atlas-ftag-tools/pull/104)
- Rename flavours to labels [#103](https://github.com/umami-hep/atlas-ftag-tools/pull/103)
- Remove docker build workflow

### [v0.2.6]

- Labeller updates for salt integration [#102](https://github.com/umami-hep/atlas-ftag-tools/pull/102)
- Add optional _px variable for ghost jets [#94](https://github.com/umami-hep/atlas-ftag-tools/pull/94)
- Added temporary discriminant for ghost jets [#99](https://github.com/umami-hep/atlas-ftag-tools/pull/99)
- Bugfix for track_selector [#98](https://github.com/umami-hep/atlas-ftag-tools/pull/98)

### [v0.2.5]

- Quick fix for add_flavour_label [#91](https://github.com/umami-hep/atlas-ftag-tools/pull/91)

### [v0.2.4]

- Add nshared selection to track selector [#90](https://github.com/umami-hep/atlas-ftag-tools/pull/90)

### [v0.2.3]

- Add track selector class based on cuts [#89](https://github.com/umami-hep/atlas-ftag-tools/pull/89)
- Add tool to generate labels on the fly [#87](https://github.com/umami-hep/atlas-ftag-tools/pull/87)

### [v0.2.2]

- Update ruff and mypy [#85](https://github.com/umami-hep/atlas-ftag-tools/pull/85)
- Fix precision setting for H5Writer [#84](https://github.com/umami-hep/atlas-ftag-tools/pull/84)
- Added new flavour for ghost association labelling [#83](https://github.com/umami-hep/atlas-ftag-tools/pull/83)

### [v0.2.1]

- Added Tau tagging discriminant [#81](https://github.com/umami-hep/atlas-ftag-tools/pull/81)
- Update labels for isolation studies [#80](https://github.com/umami-hep/atlas-ftag-tools/pull/80)
- Add cli utils module [#78](https://github.com/umami-hep/atlas-ftag-tools/pull/78)


### [v0.2.0]

- Add CLI utils [#78](https://github.com/umami-hep/atlas-ftag-tools/pull/78)

### [v0.1.19]

- Update ruff and add more rules [#77](https://github.com/umami-hep/atlas-ftag-tools/pull/77)
- Always enable ptau output in mock file generation [#76](https://github.com/umami-hep/atlas-ftag-tools/pull/76)

### [v0.1.18]

- Hotfix for Flavour frac_str

### [v0.1.17]

- Improve ftau support [#75](https://github.com/umami-hep/atlas-ftag-tools/pull/75)
- Remove print statement in `vds.py`
- Add option for tau outputs to b- and c-tagging discriminants [#74](https://github.com/umami-hep/atlas-ftag-tools/pull/74)

### [v0.1.16.1]

- Re-release of broken v0.1.16

### [v0.1.16]

- Merge only common groups in virtual datasets [#63](https://github.com/umami-hep/atlas-ftag-tools/pull/63)
- Bugfix of `get_discriminant` function [#70](https://github.com/umami-hep/atlas-ftag-tools/pull/70)

### [v0.1.15]

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
