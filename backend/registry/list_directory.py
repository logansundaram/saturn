from langchain.tools import tool
from pathlib import Path


WORKSPACE_DIR = Path("database/workspace").resolve()


@tool
def list_directory(directory_path: str = "."):
    """Lists the contents of a directory inside the workspace."""

    target_path = (WORKSPACE_DIR / directory_path).resolve()

    if not str(target_path).startswith(str(WORKSPACE_DIR)):
        return "Invalid directory path."

    if not target_path.is_dir():
        return "Path is not a directory."

    return [item.name for item in target_path.iterdir()]
