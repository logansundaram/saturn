import time
from langchain.tools import tool
from pathlib import Path


WORKSPACE_DIR = Path("database/workspace").resolve()


@tool
def read_file(file_path: str):
    """Reads the content of a file and returns it as a string. Takes the file path as an input parameter. The file path should be a string. The file path should be a valid path to a file. The file path should be a relative path from the current working directory. The file path should be a string. The file path should be a valid path to a file."""
    start = time.perf_counter()
    try:
        target_path = (WORKSPACE_DIR / file_path).resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return "Invalid file path."
        with open(target_path, "r") as file:
            content = file.read()
        return content
    finally:
        print(f"read_file : {time.perf_counter() - start:.4f}s")
