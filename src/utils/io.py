from pathlib import Path
import tarfile

from tqdm import tqdm


def tar_folder(
    src_folder: Path, dest_path: Path, regex: str = "*.png", desc="Tarring files"
) -> None:
    """
    Tar folder

    Args:
        src_folder (Path): Source folder
        dest_path (Path): Destination path
        regex (str, optional): Regex. Defaults to "*.png".
        desc (str, optional): Description. Defaults to "Tarring files".
    """
    with tarfile.open(dest_path, "w") as tar:
        for file_path in tqdm(src_folder.glob(regex), desc=desc):
            tar.add(file_path, arcname=file_path.name)
