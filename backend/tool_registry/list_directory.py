import time
from langchain.tools import tool
from pathlib import Path


WORKSPACE_DIR = Path("database/workspace").resolve()


@tool
def list_directory(directory: str = "."):
    """Lists the files and folders inside a workspace directory. directory is a path relative to the workspace root. Use '.' to list the workspace root."""
    start = time.perf_counter()
    try:
        target_path = (WORKSPACE_DIR / directory).resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return "Invalid directory path."

        if not target_path.is_dir():
            return "Path is not a directory."

        return [item.name for item in target_path.iterdir()]
    finally:
        print(f"list_directory : {time.perf_counter() - start:.4f}s")
