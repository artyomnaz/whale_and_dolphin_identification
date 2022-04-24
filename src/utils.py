from pathlib import Path


def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"
