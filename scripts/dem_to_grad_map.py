from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import get_data_window, transform as win_transform


def main():
    ref_tif_path = Path("data/raw/Greenland26X_22W_Sentinel2_2019-06-03_05.tif")
    res = None
    dst_crs = None
    dst_transform = None
    with rasterio.open(ref_tif_path, "r") as ref:
        res = ref.res[0]
        dst_crs = ref.crs
        dst_transform = ref.transform

    dem_paths = [
        Path("data/raw/10m_v4.1_dem_regions_1-4_clipped.tif"),
        Path("data/raw/10m_v4.1_dem_regions_5-6_clipped.tif"),
    ]

    dst_dem_paths = [
        Path("data/raw/38m_v4.1_dem_regions_1-4_clipped_grad.tif"),
        Path("data/raw/38m_v4.1_dem_regions_5-6_clipped_grad.tif"),
    ]

    for dem_path, dst_dem_path in zip(dem_paths, dst_dem_paths):
        print(f"Processing {dem_path}...")
        # 1. Reproject DEM
        with rasterio.open(dem_path, "r") as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(dst_dem_path, "w", **kwargs) as dst:
                reproject(
                    rasterio.band(src, 1),
                    rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                    num_threads=16,
                )

        # 2. Create gradient map from DEM
        with rasterio.open(dst_dem_path, "r") as src:
            # Get rid of nodata
            kwargs = src.meta.copy()
            data_window = get_data_window(src.read(masked=True))
            data_transform = win_transform(data_window, src.transform)
            kwargs.update(
                {
                    "height": data_window.height,
                    "width": data_window.width,
                    "transform": data_transform,
                }
            )
            dem_arr = src.read(1, window=data_window)

            grad = np.gradient(dem_arr)
            mag = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
            mag = np.where(dem_arr == -9999, mag.max(), mag)
            # clip
            mag = np.clip(mag, 0, 1)
            # scale to 0-1
            mag = (mag - mag.min()) / (mag.max() - mag.min())
            # invert
            mag = 1 - mag

            kwargs.update(
                {
                    "dtype": "float32",
                    "nodata": None,
                }
            )

            with rasterio.open(dst_dem_path, "w", **kwargs) as dst:
                dst.write(mag, 1)

        # 3. Downsample gradient map to match reference resolution
        dst_dem_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_dem_path, "r") as src:
            scale_factor_x = src.res[0] / res
            scale_factor_y = src.res[1] / res

            dem = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor_y),
                    int(src.width * scale_factor_x),
                ),
                resampling=rasterio.warp.Resampling.bilinear,
            )
            # scale image transform
            transform = src.transform * src.transform.scale(
                (1 / scale_factor_x), (1 / scale_factor_y)
            )
            kwargs.update(
                {
                    "height": dem.shape[-2],
                    "width": dem.shape[-1],
                    "transform": transform,
                }
            )

            with rasterio.open(dst_dem_path, "w", **kwargs) as dst:
                dst.write(dem)


if __name__ == "__main__":
    main()
