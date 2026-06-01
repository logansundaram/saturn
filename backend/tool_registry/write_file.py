import time
from langchain.tools import tool
from pathlib import Path

from document_registry import register_workspace_file

WORKSPACE_DIR = Path("database/workspace").resolve()


@tool
def write_file(file_path: str, content: str, overwrite: bool = False):
    """Writes content to a file in the workspace. file_path is relative to the workspace root. content is the text to write. overwrite=True replaces the file; overwrite=False (default) appends to it."""
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
