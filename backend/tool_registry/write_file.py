import time
from langchain.tools import tool
from pathlib import Path

from document_registry import register_workspace_file

WORKSPACE_DIR = Path("database/workspace").resolve()


@tool
def write_file(file_path: str, content: str, overwrite: bool = False):
    """Writes the given content to a file at the specified file path. Takes the file path, content, and operation as input parameters. Should be a string The file path should be a string. The file path should be a valid path to a file."""
    start = time.perf_counter()
    try:
        target_path = (WORKSPACE_DIR / file_path).resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return "Invalid file path."
        if overwrite:
            with open(target_path, "w") as file:
                file.write(content)
            register_workspace_file(file_path, content)
            return "File overwritten successfully"
        else:
            with open(target_path, "a") as file:
                file.write(content)
            # Read back the full file content so the manifest reflects the complete document.
            full_content = target_path.read_text(encoding="utf-8")
            register_workspace_file(file_path, full_content)
            return "Content appended to file successfully"
    finally:
        print(f"write_file : {time.perf_counter() - start:.4f}s")
