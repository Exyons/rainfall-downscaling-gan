try:
    from .constants import CHIRPS_REFERENCE_FILE_PATH, ERA5_REFERENCE_FILE_PATH, CHIRPS_GEOTIFF_DIR, ERA5_GEOTIFF_DIR
except Exception:
    from constants import CHIRPS_REFERENCE_FILE_PATH, ERA5_REFERENCE_FILE_PATH, CHIRPS_GEOTIFF_DIR, ERA5_GEOTIFF_DIR

import os
import glob
from datetime import datetime
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from multiprocessing import Pool
from tqdm import tqdm
import itertools


def fill_nan(data):
    data[np.isnan(data)] = 0.0
    return data


def reproject_data(ref_meta, src, channel):
    reprojection = np.empty(ref_meta["shape"], dtype=ref_meta["dtype"])
    reproject(
        source=rasterio.band(src, channel),
        destination=reprojection,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        dst_nodata=ref_meta["nodata"],
        resampling=Resampling.cubic,
    )
    return reprojection


# Module-level worker functions (picklable)
def load_chirps_geotiff(file_path):
    try:
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[-1].replace(".tif", "")
        date_key = datetime.strptime(date_str, "%Y-%m-%d").date()
        with rasterio.open(file_path) as src:
            data = src.read(1)
            data = fill_nan(data)
        return date_key, data, file_path
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, None, file_path
    
def load_chirps_geotiff_and_reproject_to_era5(file_path, ref_meta):
    try:
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[-1].replace(".tif", "")
        date_key = datetime.strptime(date_str, "%Y-%m-%d").date()
        with rasterio.open(file_path) as src:
            data = reproject_data(ref_meta=ref_meta, src=src, channel=1)
            data = fill_nan(data)
        return date_key, data, file_path
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, None, file_path


def load_era5_geotiff(file_path, ref_meta):
    # ref_meta is a plain dict with shape/transform/crs/dtype/nodata
    try:
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[-1].replace(".tif", "")
        date_key = datetime.strptime(date_str, "%Y-%m-%d").date()
        with rasterio.open(file_path) as src:
            air_temperature = reproject_data(ref_meta=ref_meta, src=src, channel=1)
            dewpoint_temperature = reproject_data(ref_meta=ref_meta, src=src, channel=4)
            total_precpitation = reproject_data(ref_meta=ref_meta, src=src, channel=5)
            u_wind = reproject_data(ref_meta=ref_meta, src=src, channel=8)
            v_wind = reproject_data(ref_meta=ref_meta, src=src, channel=9)
            reprojected_era5 = np.stack(
                (
                    air_temperature,
                    dewpoint_temperature,
                    total_precpitation * 1000,
                    u_wind,
                    v_wind,
                )
            )
            original_era5 = np.stack(
                (src.read(1), src.read(4), src.read(5) * 1000, src.read(8), src.read(9))
            )
        return (
            date_key,
            original_era5,
            reprojected_era5,
            file_path,
        )  # ERA5 has m/d unit, multiplying by 1000 will give mm/d. Chirps has mm/d
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, None, None, file_path


class RainfallDataLoader:
    def __init__(self) -> None:
        # don't keep an open rasterio.DatasetReader on self
        # open reference only to extract metadata, then close
        with rasterio.open(CHIRPS_REFERENCE_FILE_PATH) as ref:
            self.ref_meta_chirps = {
                "shape": ref.shape,
                "dtype": ref.dtypes[0],
                "transform": ref.transform,
                "crs": ref.crs,
                "nodata": ref.nodata,
            }
        with rasterio.open(ERA5_REFERENCE_FILE_PATH) as ref:
            self.ref_meta_era5 = {
                "shape": ref.shape,
                "dtype": ref.dtypes[0],
                "transform": ref.transform,
                "crs": ref.crs,
                "nodata": ref.nodata,
            }

        self.chirps_geotiff_files = glob.glob(
            os.path.join(CHIRPS_GEOTIFF_DIR, "CHIRPS_daily_*.tif")
        )
        self.era5_geotiff_files = glob.glob(
            os.path.join(ERA5_GEOTIFF_DIR, "ERA5_daily_*.tif")
        )

    def get_chirps_data(self, reproject_to_era5=False):
        with Pool() as pool:
            if reproject_to_era5:
                args_iter = zip(self.chirps_geotiff_files, itertools.repeat(self.ref_meta_era5))
                chirps_results = list(
                    tqdm(
                        pool.starmap(load_chirps_geotiff_and_reproject_to_era5, args_iter),
                        total=len(self.era5_geotiff_files),
                        desc="Reprojecting and Loading CHIRPS Data",
                        colour="BLUE",
                    )
                )
            else:
                chirps_results = list(
                    tqdm(
                        pool.imap(load_chirps_geotiff, self.chirps_geotiff_files),
                        total=len(self.chirps_geotiff_files),
                        desc="Loading CHIRPS Data",
                        colour="BLUE",
                    )
                )

        chirps_data_dict = {}
        for date_key, data, file_path in chirps_results:
            if date_key is not None and data is not None:
                chirps_data_dict[date_key] = data
            else:
                print(f"Skipped {file_path} due to loading error")
        return dict(sorted(chirps_data_dict.items()))

    def get_era5_data(self):
        with Pool() as pool:
            # repeat the small ref_meta dict for each task (picklable)
            args_iter = zip(self.era5_geotiff_files, itertools.repeat(self.ref_meta_chirps))
            era5_results = list(
                tqdm(
                    pool.starmap(load_era5_geotiff, args_iter),
                    total=len(self.era5_geotiff_files),
                    desc="Loading ERA5 Data",
                    colour="BLUE",
                )
            )

        original_era5_data_dict = {}
        reprojected_era5_data_dict = {}
        for date_key, original_era5, reprojected_era5, file_path in era5_results:
            if date_key is not None and (
                original_era5 is not None or reprojected_era5 is not None
            ):
                original_era5_data_dict[date_key] = original_era5
                reprojected_era5_data_dict[date_key] = reprojected_era5
            else:
                print(f"Skipped {file_path} due to loading error")

        return dict(sorted(original_era5_data_dict.items())), dict(
            sorted(reprojected_era5_data_dict.items())
        )
