from pathlib import Path
import time
import urllib.error
from urllib.request import urlretrieve
from tqdm import tqdm


TILES = [
    "15_40",
    "17_40",
    "16_40",
    "14_39",
    "18_39",
    "29_45",
    "17_39",
    "16_39",
    "17_38",
    "15_39",
    "30_45",
    "13_39",
    "31_44",
    "30_43",
    "29_46",
    "31_45",
    "27_44",
    "14_40",
    "13_40",
    "15_38",
    "30_44",
    "29_44",
    "14_38",
    "28_43",
    "16_38",
    "13_38",
    "18_40",
    "27_45",
    "28_45",
    "29_43",
    "28_46",
    "28_44",
]
TILE_RESOLUTION = "10m"  # 32m or 10m


def update_progress(blocknum: int, blocksize: int, totalsize: int, progress):
    """Helper function to update progress bar."""
    if progress.total != totalsize:
        progress.total = totalsize
        progress.refresh()
    progress.update(blocknum * blocksize - progress.n)


def download_tile(tile: str, size: str, retry: int = 0, max_reties: int = 10):
    """Download a tile from ArcticDEM. Supports only 32m or 10m resolution."""
    dir = Path(f"tiles_{size}")
    dir.mkdir(exist_ok=True)

    progress = tqdm(
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {tile} {size}",
        position=1,
        leave=True,
    )
    try:
        urlretrieve(
            f"https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v4.1/{size}/{tile}/{tile}_{size}_v4.1.tar.gz",
            dir / f"{tile}_{size}_v4.1.tar.gz",
            reporthook=lambda blocknum, blocksize, totalsize: update_progress(
                blocknum, blocksize, totalsize, progress
            ),
        )
    except urllib.error.URLError:
        if max_reties > retry:
            time.sleep(15)
            download_tile(tile, retry + 1, max_reties)
        else:
            print(f"Failed to download {tile} {size}")


def download_tiles(tiles: list[str]):
    """Download a list of tiles."""
    for tile in tqdm(tiles, desc="Downloading tiles"):
        download_tile(tile, TILE_RESOLUTION)


if __name__ == "__main__":
    """
    Download tiles from ArcticDEM.

    Usage:
        > nohup python3 download_articdem.py > out.log 2>&1 &
    """
    download_tiles(TILES)
