import typing as T
from pathlib import Path
from functools import partial

from .utils import LabeledData, get_image_list_dims
from ..augment.augmentation import augment
from ..errors import TopologyClipError
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import TqdmParallel

import geowombat as gw
from geowombat.core import polygon_to_array
from geowombat.core.windows import get_window_offsets
import numpy as np
from scipy.ndimage.measurements import label as nd_label
import cv2
from rasterio.windows import Window
import xarray as xr
import geopandas as gpd
from skimage.measure import regionprops
from tqdm.auto import tqdm
from torch_geometric.data import Data
import joblib
from joblib import delayed, parallel_backend


logger = set_color_logger(__name__)


def roll(
    arr_pad: np.ndarray,
    shift: T.Union[int, T.Tuple[int, int]],
    axis: T.Union[int, T.Tuple[int, int]]
) -> np.ndarray:
    """Rolls array elements along a given axis and slices off padded edges"""
    return np.roll(arr_pad, shift, axis=axis)[1:-1, 1:-1]


def close_edge_ends(array: np.ndarray) -> np.ndarray:
    """Closes 1 pixel gaps at image edges
    """
    # Top
    idx = np.where(array[1] == 1)
    z = np.zeros(array.shape[1], dtype='uint8')
    z[idx] = 1
    array[0] = z
    # Bottom
    idx = np.where(array[-2] == 1)
    z = np.zeros(array.shape[1], dtype='uint8')
    z[idx] = 1
    array[-1] = z
    # Left
    idx = np.where(array[:, 1] == 1)
    z = np.zeros(array.shape[0], dtype='uint8')
    z[idx] = 1
    array[:, 0] = z
    # Right
    idx = np.where(array[:, -2] == 1)
    z = np.zeros(array.shape[0], dtype='uint8')
    z[idx] = 1
    array[:, -1] = z

    return array


def get_other_crop_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    rarray = roll(array_pad, 1, axis=0)
    crop_count = np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, -1, axis=0)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, 1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, -1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))

    return crop_count


def fill_edge_gaps(labels: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fills neighboring 1-pixel edge gaps
    """
    # array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')
    # hsum = roll(array_pad, 1, axis=0) + roll(array_pad, -1, axis=0)
    # vsum = roll(array_pad, 1, axis=1) + roll(array_pad, -1, axis=1)
    # array = np.where(
    #     (hsum == 2) & (vsum == 0), 1, array
    # )
    # array = np.where(
    #     (hsum == 0) & (vsum == 2), 1, array
    # )
    other_count = get_other_crop_count(np.where(array == 1, 0, labels))
    array = np.where(
        other_count > 0, 1, array
    )

    return array


def get_crop_count(array: np.ndarray, edge_class: int) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    rarray = roll(array_pad, 1, axis=0)
    crop_count = np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, -1, axis=0)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, 1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, -1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))

    return crop_count


def get_edge_count(array: np.ndarray, edge_class: int) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    edge_count = np.uint8(roll(array_pad, 1, axis=0) == edge_class)
    edge_count += np.uint8(roll(array_pad, -1, axis=0) == edge_class)
    edge_count += np.uint8(roll(array_pad, 1, axis=1) == edge_class)
    edge_count += np.uint8(roll(array_pad, -1, axis=1) == edge_class)

    return edge_count


def get_non_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    non_count = np.uint8(roll(array_pad, 1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, 1, axis=1) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=1) == 0)

    return non_count


def cleanup_edges(array: np.ndarray, original: np.ndarray, edge_class: int) -> np.ndarray:
    """Removes crop pixels that border non-crop pixels
    """
    array_pad = np.pad(original, pad_width=((1, 1), (1, 1)), mode='edge')
    original_zero = np.uint8(roll(array_pad, 1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, 1, axis=1) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=1) == 0)

    # Fill edges
    array = np.where(
        (array == 0) & (get_crop_count(array, edge_class) > 0) & (get_edge_count(array, edge_class) > 0),
        edge_class, array
    )
    # Remove crops next to non-crop
    array = np.where(
        (array > 0) & (array != edge_class) & (get_non_count(array) > 0) & (get_edge_count(array, edge_class) > 0),
        0, array
    )
    # Fill in non-cropland
    array = np.where(
        original_zero == 4, 0, array
    )
    # Remove isolated crop pixels (i.e., crop clumps with 2 or fewer pixels)
    array = np.where(
        (array > 0) & (array != edge_class) & (get_crop_count(array, edge_class) <= 1), 0, array
    )

    return array


def is_grid_processed(
    process_path: Path,
    transforms: T.List[str],
    group_id: str,
    grid: T.Union[str, int],
    n_ts: int
) -> bool:
    """Checks if a grid is already processed
    """
    batch_stored = False
    for aug in transforms:
        if aug.startswith('ts-'):
            for i in range(0, n_ts):
                train_id = f'{group_id}_{grid}_{aug}_{i:03d}'
                train_path = process_path / f'data_{train_id}.pt'
                if train_path.is_file():
                    train_data = joblib.load(train_path)
                    if train_data.train_id == train_id:
                        batch_stored = True

        else:
            train_id = f'{group_id}_{grid}_{aug}'
            train_path = process_path / f'data_{train_id}.pt'
            if train_path.is_file():
                train_data = joblib.load(train_path)
                if train_data.train_id == train_id:
                    batch_stored = True

    return batch_stored


def create_boundary_distances(
    labels_array: np.ndarray, train_type: str, cell_res: float
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates distances from boundaries
    """
    if train_type.lower() == 'polygon':
        mask = np.uint8(labels_array)
    else:
        mask = np.uint8(1 - labels_array)
    # Get unique segments
    segments = nd_label(mask)[0]
    # Get the distance from edges
    bdist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        3
    )
    bdist *= cell_res

    grad_x = cv2.Sobel(
        np.pad(bdist, 5, mode='edge'),
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=5
    )
    grad_y = cv2.Sobel(
        np.pad(bdist, 5, mode='edge'),
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=5
    )
    ori = cv2.phase(grad_x, grad_y, angleInDegrees=False)
    ori = ori[5:-5, 5:-5] / np.deg2rad(360)
    ori[labels_array == 0] = 0

    return mask, segments, bdist, ori


def normalize_boundary_distances(
    labels_array: np.ndarray,
    train_type: str,
    cell_res: float,
    normalize: bool = True
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Normalizes boundary distances
    """
    # Create the boundary distances
    __, segments, bdist, ori = create_boundary_distances(labels_array, train_type, cell_res)
    dist_max = 1e9
    if normalize:
        dist_max = 1.0
        # Normalize each segment by the local max distance
        props = regionprops(segments, intensity_image=bdist)
        for p in props:
            if p.label > 0:
                bdist = np.where(
                    segments == p.label,
                    bdist / p.max_intensity,
                    bdist
                )
    bdist = np.nan_to_num(
        bdist.clip(0, dist_max),
        nan=1.0,
        neginf=1.0,
        posinf=1.0
    )
    ori = np.nan_to_num(
        ori.clip(0, 1),
        nan=1.0,
        neginf=1.0,
        posinf=1.0
    )

    return bdist, ori


def edge_gradient(array: np.ndarray) -> np.ndarray:
    """Calculates the morphological gradient of crop fields
    """
    se = np.array(
        [
            [1, 1],
            [1, 1]
        ], dtype='uint8'
    )
    array = np.uint8(cv2.morphologyEx(np.uint8(array), cv2.MORPH_GRADIENT, se) > 0)

    return array


def create_image_vars(
    image: T.Union[str, Path, list],
    max_crop_class: int,
    bounds: tuple,
    num_workers: int,
    gain: float = 1e-4,
    offset: float = 0.0,
    grid_edges: T.Optional[gpd.GeoDataFrame] = None,
    ref_res: T.Optional[float] = 10.0,
    resampling: T.Optional[str] = 'nearest',
    crop_column: T.Optional[str] = 'class',
    keep_crop_classes: T.Optional[bool] = False,
    replace_dict: T.Optional[T.Dict[int, int]] = None
) -> T.Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Creates the initial image training data
    """
    edge_class = max_crop_class + 1

    if isinstance(image, list):
        image = [str(fn) for fn in image]

    # Open the image variables
    with gw.config.update(ref_bounds=bounds, ref_res=ref_res):
        with gw.open(
            image,
            stack_dim='band',
            band_names=list(range(1, len(image) + 1)),
            resampling=resampling
        ) as src_ts:
            # 65535 'no data' values = nan
            mask = xr.where(src_ts > 10_000, np.nan, 1)
            # X variables
            time_series = (
                src_ts.gw.set_nodata(
                    src_ts.gw.nodataval, 0,
                    out_range=(0, 1),
                    dtype='float64',
                    scale_factor=gain,
                    offset=offset
                ) * mask
            ).fillna(0).gw.compute(num_workers=num_workers)

            # Get the time and band count
            ntime, nbands = get_image_list_dims(image, src_ts)
            if grid_edges is not None:
                if replace_dict is not None:
                    for crop_class in grid_edges[crop_column].unique():
                        if crop_class not in list(replace_dict.keys()):
                            grid_edges[crop_column] = grid_edges[crop_column].replace({crop_class: -999})
                    replace_dict[-999] = 1
                    grid_edges[crop_column] = grid_edges[crop_column].replace(replace_dict)
                    # Remove any non-crop polygons
                    grid_edges = grid_edges.query(f"{crop_column} != 0")
                if grid_edges.empty:
                    labels_array = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')
                    bdist = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                    ori = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                    edges = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')
                else:
                    # Get the field polygons
                    labels_array_copy = polygon_to_array(
                        grid_edges.assign(
                            **{
                                crop_column: range(1, len(grid_edges.index)+1)
                            }
                        ),
                        col=crop_column,
                        data=src_ts,
                        all_touched=False
                    ).squeeze().gw.compute(num_workers=num_workers)
                    labels_array = polygon_to_array(
                        grid_edges,
                        col=crop_column,
                        data=src_ts,
                        all_touched=False
                    ).squeeze().gw.compute(num_workers=num_workers)
                    # Get the field edges
                    edges = polygon_to_array(
                        (
                            grid_edges
                            .boundary
                            .to_frame(name='geometry')
                            .reset_index()
                            .rename(columns={'index': crop_column})
                            .assign(**{crop_column: range(1, len(grid_edges.index)+1)})
                        ),
                        col=crop_column,
                        data=src_ts,
                        all_touched=False
                    ).squeeze().gw.compute(num_workers=num_workers)
                    edges[edges > 0] = 1
                    assert edges.max() <= 1, 'Edges were not created.'
                    if edges.max() == 0:
                        return None, None, None, None, None, None
                    image_grad = edge_gradient(labels_array_copy)
                    image_grad_count = get_crop_count(image_grad, edge_class)
                    edges = np.where(image_grad_count > 0, edges, 0)
                    # Recode
                    if not keep_crop_classes:
                        labels_array = np.where(
                            labels_array > 0, max_crop_class, 0
                        )
                    # Set edges
                    labels_array[edges == 1] = edge_class
                    # No crop pixel should border non-crop
                    labels_array = cleanup_edges(labels_array, labels_array_copy, edge_class)
                    assert labels_array.max() <= edge_class, \
                        'The labels array have larger than expected values.'
                    # Normalize the boundary distances for each segment
                    bdist, ori = normalize_boundary_distances(
                        np.uint8((labels_array > 0) & (labels_array != edge_class)),
                        grid_edges.geom_type.values[0],
                        src_ts.gw.celly
                    )
                # import matplotlib.pyplot as plt
                # def save_labels(out_fig: Path):
                #     fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True, dpi=300)
                #     axes = axes.flatten()
                #     for ax, im, title in zip(
                #         axes,
                #         (labels_array_copy, labels_array, bdist, ori),
                #         ('Fields', 'Edges', 'Distance', 'Orientation')
                #     ):
                #         ax.imshow(im, interpolation='nearest')
                #         ax.set_title(title)
                #         ax.axis('off')

                #     plt.tight_layout()
                #     plt.savefig(out_fig, dpi=300)
                # import uuid
                # fig_dir = Path('figures')
                # fig_dir.mkdir(exist_ok=True, parents=True)
                # hash_id = uuid.uuid4().hex
                # save_labels(
                #     out_fig=fig_dir / f'{hash_id}.png'
                # )
            else:
                labels_array = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')
                bdist = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                ori = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                edges = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')

    return time_series, labels_array, bdist, ori, ntime, nbands


def save_and_update(
    write_path: Path,
    predict_data: Data,
    name: str,
    compress: int = 5
) -> None:
    predict_path = write_path / f'data_{name}.pt'
    joblib.dump(
        predict_data,
        predict_path,
        compress=compress
    )


def read_slice(darray: xr.DataArray, w_pad: Window):
    slicer = (
        slice(0, None),
        slice(w_pad.row_off, w_pad.row_off+w_pad.height),
        slice(w_pad.col_off, w_pad.col_off+w_pad.width)
    )

    return darray[slicer].gw.compute()


def get_window_chunk(
    windows: T.List[T.Tuple[Window, Window]],
    chunksize: int
) -> T.List[T.Tuple[Window, Window]]:
    for i in range(0, len(windows), chunksize):
        yield windows[i:i+chunksize]


def create_and_save_window(
    write_path: Path,
    ntime: int,
    nbands: int,
    image_height: int,
    image_width: int,
    res: float,
    resampling: str,
    region: str,
    year: int,
    window_size: int,
    padding: int,
    x: np.ndarray,
    w: Window,
    w_pad: Window,
) -> None:
    size = window_size + padding*2
    x_height = x.shape[1]
    x_width = x.shape[2]

    row_pad_before = 0
    col_pad_before = 0
    col_pad_after = 0
    row_pad_after = 0
    if (x_height != size) or (x_width != size):
        # Pre-padding
        if w.row_off < padding:
            row_pad_before = padding - w.row_off
        if w.col_off < padding:
            col_pad_before = padding - w.col_off
        # Post-padding
        if w.row_off + window_size + padding > image_height:
            row_pad_after = size - x_height
        if w.col_off + window_size + padding > image_width:
            col_pad_after = size - x_width

        x = np.pad(
            x,
            pad_width=(
                (0, 0),
                (row_pad_before, row_pad_after),
                (col_pad_before, col_pad_after)
            ),
            mode='constant',
        )
    if x.shape[1:] != (size, size):
        logger.exception('The array does not match the expected size.')

    ldata = LabeledData(
        x=x,
        y=None,
        bdist=None,
        ori=None,
        segments=None,
        props=None
    )
    predict_data = augment(
        ldata,
        aug='none',
        ntime=ntime,
        nbands=nbands,
        max_crop_class=0,
        k=3,
        instance_seg=False,
        zero_padding=0,
        window_row_off=w.row_off,
        window_col_off=w.col_off,
        window_height=w.height,
        window_width=w.width,
        window_pad_row_off=w_pad.row_off,
        window_pad_col_off=w_pad.col_off,
        window_pad_height=w_pad.height,
        window_pad_width=w_pad.width,
        row_pad_before=row_pad_before,
        row_pad_after=row_pad_after,
        col_pad_before=col_pad_before,
        col_pad_after=col_pad_after,
        res=res,
        resampling=resampling
    )
    save_and_update(
        write_path, predict_data, f'{region}_{year}_{w.row_off}_{w.col_off}'
    )


def create_predict_dataset(
    image_list: T.List[T.List[T.Union[str, Path]]],
    region: str,
    year: int,
    process_path: Path = None,
    gain: float = 1e-4,
    offset: float = 0.0,
    ref_res: float = 10.0,
    resampling: str = 'nearest',
    window_size: int = 100,
    padding: int = 101,
    num_workers: int = 1,
    chunksize: int = 100
):
    with gw.config.update(ref_res=ref_res):
        with gw.open(
            image_list,
            stack_dim='band',
            band_names=list(range(1, len(image_list) + 1)),
            resampling=resampling,
            chunks=512
        ) as src_ts:
            windows = get_window_offsets(
                src_ts.gw.nrows,
                src_ts.gw.ncols,
                window_size,
                window_size,
                padding=(
                    padding, padding, padding, padding
                )
            )
            time_series = (
                (src_ts.astype('float64') * gain + offset)
                .clip(0, 1)
            )

            ntime, nbands = get_image_list_dims(image_list, src_ts)
            partial_create = partial(
                create_and_save_window,
                process_path,
                ntime,
                nbands,
                src_ts.gw.nrows,
                src_ts.gw.ncols,
                ref_res,
                resampling,
                region,
                year,
                window_size,
                padding
            )

            with tqdm(
                total=len(windows),
                desc='Creating prediction windows',
                position=0
            ) as pbar_total:
                with parallel_backend(
                    backend='loky',
                    n_jobs=num_workers
                ):
                    for window_chunk in get_window_chunk(windows, chunksize):
                        with TqdmParallel(
                            tqdm_kwargs={
                                'total': len(window_chunk),
                                'desc': 'Window chunks',
                                'position': 1,
                                'leave': False
                            },
                            temp_folder='/tmp'
                        ) as pool:
                            __ = pool(
                                delayed(partial_create)(
                                    read_slice(time_series, window_pad), window, window_pad
                                ) for window, window_pad in window_chunk
                            )
                        pbar_total.update(len(window_chunk))


def create_dataset(
    image_list: T.List[T.List[T.Union[str, Path]]],
    df_grids: gpd.GeoDataFrame,
    df_edges: gpd.GeoDataFrame,
    max_crop_class: int,
    group_id: str = None,
    process_path: Path = None,
    transforms: T.List[str] = None,
    gain: float = 1e-4,
    offset: float = 0.0,
    ref_res: float = 10.0,
    resampling: str = 'nearest',
    num_workers: int = 1,
    grid_size: T.Optional[T.Union[T.Tuple[int, int], T.List[int], None]] = None,
    n_ts: T.Optional[int] = 2,
    instance_seg: T.Optional[bool] = False,
    zero_padding: T.Optional[int] = 0,
    crop_column: T.Optional[str] = 'class',
    keep_crop_classes: T.Optional[bool] = False,
    replace_dict: T.Optional[T.Dict[int, int]] = None
) -> None:
    """Creates a dataset for training

    Args:
        image_list: A list of images.
        df_grids: The training grids.
        df_edges: The training edges.
        max_crop_class: The maximum expected crop class value.
        group_id: A group identifier, used for logging.
        process_path: The main processing path.
        transforms: A list of augmentation transforms to apply.
        gain: A gain factor to apply to the images.
        offset: An offset factor to apply to the images.
        ref_res: The reference cell resolution to resample the images to.
        resampling: The image resampling method.
        num_workers: The number of dask workers.
        grid_size: The requested grid size, in (rows, columns) or (height, width).
        lc_path: The land cover image path.
        n_ts: The number of temporal augmentations.
        data_type: The target data type.
        instance_seg: Whether to get instance segmentation mask targets.
        zero_padding: Zero padding to apply.
        crop_column: The crop column name in the polygon vector files.
        keep_crop_classes: Whether to keep the crop classes as they are (True) or recode all
            non-zero classes to crop (False).
        replace_dict: A dictionary of crop class remappings.
    """
    if transforms is None:
        transforms = ['none']

    merged_grids = []
    sindex = df_grids.sindex

    # Get the image CRS
    with gw.open(image_list[0]) as src:
        image_crs = src.crs

    with tqdm(total=df_grids.shape[0], desc='Check') as pbar:
        for row in df_grids.itertuples():
            # Clip the edges to the current grid
            try:
                grid_edges = gpd.clip(df_edges, row.geometry)
            except:
                logger.warning(
                    TopologyClipError('The input GeoDataFrame contains topology errors.')
                )
                df_edges = gpd.GeoDataFrame(
                    data=df_edges[crop_column].values,
                    columns=[crop_column],
                    geometry=df_edges.buffer(0).geometry
                )
                grid_edges = gpd.clip(df_edges, row.geometry)

            # These are grids with no crop fields. They should still
            # be used for training.
            if grid_edges.loc[~grid_edges.is_empty].empty:
                grid_edges = df_grids.copy()
                grid_edges = grid_edges.assign(**{crop_column: 0})
            # Remove empty geometry
            grid_edges = grid_edges.loc[~grid_edges.is_empty]

            if not grid_edges.empty:
                # Check if the edges overlap multiple grids
                int_idx = sorted(
                    list(
                        sindex.intersection(
                            tuple(grid_edges.total_bounds.flatten())
                        )
                    )
                )

                if len(int_idx) > 1:
                    # Check if any of the grids have already been stored
                    if any(
                        [
                            rowg in merged_grids for rowg in df_grids.iloc[int_idx].grid.values.tolist()
                        ]
                    ):
                        pbar.update(1)
                        pbar.set_description(f'No edges for {group_id}')
                        continue

                    grid_edges = gpd.clip(df_edges, df_grids.iloc[int_idx].geometry)
                    merged_grids.append(row.grid)

                nonzero_mask = grid_edges[crop_column] != 0

                # left, bottom, right, top
                ref_bounds = df_grids.to_crs(image_crs).iloc[int_idx].total_bounds.tolist()
                if grid_size is not None:
                    height, width = grid_size
                    left, bottom, right, top = ref_bounds
                    ref_bounds = [left, top-ref_res*height, left+ref_res*width, top]

                # Data for graph network
                xvars, labels_array, bdist, ori, ntime, nbands = create_image_vars(
                    image=image_list,
                    max_crop_class=max_crop_class,
                    bounds=ref_bounds,
                    num_workers=num_workers,
                    gain=gain,
                    offset=offset,
                    grid_edges=grid_edges if nonzero_mask.any() else None,
                    ref_res=ref_res,
                    resampling=resampling,
                    crop_column=crop_column,
                    keep_crop_classes=keep_crop_classes,
                    replace_dict=replace_dict
                )
                if xvars is None:
                    pbar.update(1)
                    pbar.set_description(f'No valid fields for {group_id}')
                    continue
                if (xvars.shape[1] < 5) or (xvars.shape[2] < 5):
                    pbar.update(1)
                    pbar.set_description(f'{group_id} is too small')
                    continue

                # Check if the grid has already been saved
                if hasattr(row, 'grid'):
                    row_grid_id = row.grid
                elif hasattr(row, 'region'):
                    row_grid_id = row.region
                else:
                    raise AttributeError("The grid id should be given as 'grid' or 'region'.")

                batch_stored = is_grid_processed(
                    process_path,
                    transforms,
                    group_id,
                    row_grid_id,
                    n_ts
                )
                if batch_stored:
                    pbar.update(1)
                    pbar.set_description(f'{group_id} is already stored.')
                    continue

                # Get the upper left lat/lon
                left, bottom, right, top = (
                    df_grids.iloc[int_idx]
                    .to_crs('epsg:4326')
                    .total_bounds
                    .tolist()
                )

                if isinstance(group_id, str):
                    end_year = int(group_id.split('_')[-1])
                    start_year = end_year - 1
                else:
                    start_year, end_year = None, None

                segments = nd_label(labels_array)[0]
                props = regionprops(segments)

                ldata = LabeledData(
                    x=xvars,
                    y=labels_array,
                    bdist=bdist,
                    ori=ori,
                    segments=segments,
                    props=props
                )
                def save_and_update(train_data: Data) -> None:
                    train_path = process_path / f'data_{train_data.train_id}.pt'
                    joblib.dump(
                        train_data,
                        train_path,
                        compress=5
                    )

                for aug in transforms:
                    if aug.startswith('ts-'):
                        for i in range(0, n_ts):
                            train_id = f'{group_id}_{row_grid_id}_{aug}_{i:03d}'
                            train_data = augment(
                                ldata,
                                aug=aug,
                                ntime=ntime,
                                nbands=nbands,
                                max_crop_class=max_crop_class,
                                k=3,
                                instance_seg=instance_seg,
                                zero_padding=zero_padding,
                                start_year=start_year,
                                end_year=end_year,
                                left=left,
                                bottom=bottom,
                                right=right,
                                top=top,
                                res=ref_res,
                                train_id=train_id
                            )
                            if instance_seg:
                                if hasattr(train_data, 'boxes') and train_data.boxes is not None:
                                    save_and_update(train_data)
                            else:
                                save_and_update(train_data)
                    else:
                        train_id = f'{group_id}_{row_grid_id}_{aug}'
                        train_data = augment(
                            ldata,
                            aug=aug,
                            ntime=ntime,
                            nbands=nbands,
                            max_crop_class=max_crop_class,
                            k=3,
                            instance_seg=instance_seg,
                            zero_padding=zero_padding,
                            start_year=start_year,
                            end_year=end_year,
                            left=left,
                            bottom=bottom,
                            right=right,
                            top=top,
                            res=ref_res,
                            train_id=train_id
                        )
                        if instance_seg:
                            # Grids without boxes are set as None, and Data() does not
                            # keep None types.
                            if hasattr(train_data, 'boxes') and train_data.boxes is not None:
                                save_and_update(train_data)
                        else:
                            save_and_update(train_data)

            pbar.update(1)
            pbar.set_description(group_id)
