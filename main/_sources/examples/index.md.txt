# Example Scripts

On the following page, examples will be given how the code can be used. The first part
covers scripts and functions that can be used directly from the terminal while the second
part gives a more detailed look on the different functionalities.

## Calculate WPs

This package contains a script to calculate tagger working points (WPs).
The script is `working_points.py` and can be run after installing this package with

```
wps \
    --ttbar "path/to/ttbar/*.h5" \
    --tagger GN2v01 \
    --fc 0.1
```

Both the `--tagger` and `--fc` options accept a list if you want to get the WPs for multiple taggers.
If you are doing c-tagging or xbb-tagging, dedicated fx arguments are available ()you can find them all with `-h`.

If you want to use the `ttbar` WPs get the efficiencies and rejections for the `zprime` sample, you can add `--zprime "path/to/zprime/*.h5"` to the command.
Note that a default selection of $p_T > 250 ~GeV$ to jets in the `zprime` sample.

If instead of defining the working points for a series of signal efficiencies, you wish to calculate a WP corresponding to a specific background rejection, the `--rejection` option can be given along with the desired background.

By default the working points are printed to the terminal, but you can save the results to a YAML file with the `--outfile` option.

See `wps --help` for more options and information.

## Calculate efficiency at discriminant cut 

The same script can be used to calculate the efficiency and rejection values at a given discriminant cut value.
The script `working_points.py` can be run after intalling this package as follows

```
wps \
    --ttbar "path/to/ttbar/*.h5" \
    --tagger GN2v01 \
    --fx 0.1
    --disc_cuts 1.0 1.5
```
The `--tagger`, `--fx`, and `--outfile` follow the same procedure as in the 'Calculate WPs' script as described above.

## H5 Utils

### Create virtual file

This package contains a script to easily merge a set of H5 files.
A virtual file is a fast and lightweight way to wrap a set of files.
See the [h5py documentation](https://docs.h5py.org/en/stable/vds.html) for more information on virtual datasets.

The script is `vds.py` and can be run after installing this package with

```
vds <pattern> <output path>
```

The `<pattern>` argument should be a quotes enclosed [glob pattern](https://en.wikipedia.org/wiki/Glob_(programming)), for example `"dsid/path/*.h5"`

See `vds --help` for more options and information.


### [h5move](ftag/hdf5/h5move.py)

A script to move/rename datasets inside an h5file.
Useful for correcting discrepancies between group names.
See [h5move.py](ftag/hdf5/h5move.py) for more info.


### [h5split](ftag/hdf5/h5split.py)

A script to split a large h5 file into several smaller files.
Useful if output files are too large for EOS/grid storage.
See [h5split.py](ftag/hdf5/h5split.py) for more info.

# Extensive Examples

The content below is generated automatically from `ftag/example.ipynb`.

```{include} example.md
:level: 2
```