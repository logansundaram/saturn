import time
from langchain.tools import tool

from config import get_config


@tool
def read_file(file_path: str):
    """Reads the contents of a file in the workspace and returns it as a string. file_path is relative to the workspace root."""
    start = time.perf_counter()
    try:
        # Resolve the workspace from config per call so a live `/config paths.workspace` change
        # is honored without a restart (config is the single source of truth).
        workspace = get_config().path("workspace")
        target_path = (workspace / file_path).resolve()

        if not target_path.is_relative_to(workspace):
            return "Invalid file path: outside the workspace."
        with open(target_path, "r") as file:
            content = file.read()
        return content
    finally:
        print(f"read_file : {time.perf_counter() - start:.4f}s")
