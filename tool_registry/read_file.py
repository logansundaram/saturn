import time
from langchain.tools import tool
from pathlib import Path


WORKSPACE_DIR = (Path(__file__).parent.parent / "database" / "workspace").resolve()


@tool
def read_file(file_path: str):
    """Reads the contents of a file in the workspace and returns it as a string. file_path is relative to the workspace root."""
    start = time.perf_counter()
    try:
        target_path = (WORKSPACE_DIR / file_path).resolve()

        if not target_path.is_relative_to(WORKSPACE_DIR):
            return "Invalid file path: outside the workspace."
        with open(target_path, "r") as file:
            content = file.read()
        return content
    finally:
        print(f"read_file : {time.perf_counter() - start:.4f}s")
