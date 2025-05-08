# Utils to take an input h5 file, and append one or more columns to it
import numpy as np
from ftag.hdf5.h5reader import H5Reader
from ftag.hdf5.h5writer import H5Writer
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import Callable
def merge_dicts(dicts):
    """
    Merges a list of dictionaries whre each dict is of the form:
     {
        group1: {
            variable_1: np.array
            variable_2: np.array
        },
        group2: {
            variable_1: np.array
            variable_2: np.array
        }
     }

     E.g.

     dict1 = {
        "jets": {
            "pt": np.array([1, 2, 3]),
            "eta": np.array([4, 5, 6])
        },
    }
    dict2 = {
        "jets": {
            "phi": np.array([7, 8, 9]),
            "energy": np.array([10, 11, 12])
        },
    }

    merged = {
        "jets": {
            "pt": np.array([1, 2, 3]),
            "eta": np.array([4, 5, 6]),
            "phi": np.array([7, 8, 9]),
            "energy": np.array([10, 11, 12])
        }
    }

    Parameters
    ----------
    dicts : list[dict[str, dict[str, np.ndarray]]]
        List of dictionaries to merge. Each dictionary should be of the form:

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Merged dictionary of the form:
        {
            group1: {
                variable_1: np.array
                variable_2: np.array
            },
            group2: {
                variable_1: np.array
                variable_2: np.array
            }
        }
    """
    merged = {}
    for d in dicts:
        for group, variables in d.items():
            if group not in merged:
                merged[group] = {}
            for variable, data in variables.items():
                if variable not in merged[group]:
                    merged[group][variable] = data
                else:
                    raise ValueError(f"Variable {variable} already exists in group {group}.")
    return merged


def get_shape(N, batch):
    """Returns a dictionary with the correct output shapes for the H5Writer

    Parameters
    ----------
    N : int
        Number of jets to write in total
    batch : dict[str, np.ndarray]
        Dictionary representing the batch

    Returns
    -------
    dict[str, tuple[int]]
        Dictionary with the shapes of the output arrays
    """
    shape = {}

    for key in batch.keys():
        if batch[key].ndim == 1:
            shape[key] = (N,)
        else:
            shape[key] = (N,) + batch[key].shape[1:]
    return shape

def get_all_groups(file):
    """Returns a dictionary with all groups in the h5 file"""   
    with h5py.File(file, "r") as f:
        groups = list(f.keys())
        return { g : None for g in groups}
    

def h5_add_column(
    input_file : str | Path, 
    output_file : str | Path, 
    append_function : Callable | list[Callable], 
    num_jets : int = -1 ,
    input_groups : list[str] | None = None,
    output_groups : list[str] | None = None,
    reader_kwargs : dict = {},
    writer_kwargs : dict = {},
    overwrite : bool = False,
    ):
    """Appends one or more columns to one or more groups in an h5 file.

    Parameters
    ----------
    input_file : str | Path
        Input h5 file to read from.
    output_file : str | Path
        Output h5 file to write to.
    append_function : callable | list[callable]
        A function, or list of functions, which take a batch from H5Reader and returns a dictionary 
        of the form:
            {
                group1 : {
                    new_column1 : data,
                    new_column2 : data,
                },
                group2 : {
                    new_column3 : data,
                    new_column4 : data,
                },
                ...
            }
    num_jets : int, optional
        Number of jets to read from the input file. If -1, reads all jets. By default -1.
    input_groups : list[str] | None, optional
        List of groups to read from the input file. If None, reads all groups. By default None.
    output_groups : list[str] | None, optional
        List of groups to write to the output file. If None, writes all groups. By default None.
        Note that this is a subset of the input groups, and must include all groups that the append fucntions
        wish to write to.
    reader_kwargs : dict, optional
        Additional arguments to pass to the H5Reader. By default {}.
    writer_kwargs : dict, optional
        Additional arguments to pass to the H5Writer. By default {}.
    overwrite : bool, optional
        If True, will overwrite the output file if it exists. By default False.
        If False, will raise a FileExistsError if the output file exists.
        If None, will check if the output file exists and raise an error if it does unless overwrite is True.
    """
    
    input_file = Path(input_file)
    output_file = Path(output_file) if output_file is not None else None

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if output_file is not None and output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_file} already exists. Please choose a different name.")
    
    if output_file is None:
        output_file = input_file.replace(".h5", "_additional.h5")

    if not isinstance(append_function, list):
        append_function = [append_function]
    
    reader = H5Reader(input_file, shuffle=False, **reader_kwargs)
    if 'precision' not in writer_kwargs:
        writer_kwargs['precision'] = 'full'
    
    njets = reader.num_jets if num_jets == -1 else num_jets
    writer = None

    input_groups = get_all_groups(input_file) if input_groups is None else {k : None for k in input_groups}
    output_groups = list(input_groups.keys()) if output_groups is None else output_groups
    
    assert all([o in input_groups for o in output_groups]), f"Output groups {output_groups} not in input groups {input_groups.keys()}"


    num_batches = njets // reader.batch_size + 1
    for i, batch in enumerate(reader.stream(input_groups, num_jets=njets)):
        if (i+1) % 10 == 0:
            print(f"Processing batch {i+1}/{num_batches} ({(i+1)/num_batches*100:.2f}%)")

        to_append = merge_dicts(
            [af(batch) for af in append_function]
        )
        for k, newvars in to_append.items():
            if k not in output_groups:
                raise ValueError(f"Trying to output to {k} but only {output_groups} are allowed")
            for newkey, newval in newvars.items():
                if newkey in batch[k].dtype.names:
                    raise ValueError(f"Trying to append {newkey} to {k} but it already exists in batch") 
                if newval.shape != batch[k].shape:
                    raise ValueError(f"Trying to append {newkey} to {k} but the shape is not correct")

        to_write = {}

        for key, str_array in batch.items():
            if key not in output_groups:
                continue
            if key in to_append:
                
                str_array = np.lib.recfunctions.append_fields(
                    str_array,
                    list(to_append[key].keys()),
                    list(to_append[key].values()),
                    usemask=False,
                )
            to_write[key] = str_array
        if writer is None:
            writer = H5Writer(
                output_file,
                dtypes={key: str_array.dtype for key, str_array in to_write.items()},
                shapes=get_shape(njets, to_write),
                shuffle=False,
                **writer_kwargs,
            )
        
        writer.write(to_write)
                
