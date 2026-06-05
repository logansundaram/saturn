import time
from langchain.tools import tool

from config import get_config
from document_registry import register_workspace_file


@tool
def write_file(file_path: str, content: str, overwrite: bool = True):
    """Writes content to a file in the workspace. file_path is relative to the workspace root. content is the text to write. overwrite=True (default) replaces the file's contents; pass overwrite=False to append to the existing file instead."""
    start = time.perf_counter()
    try:
        # Resolve the workspace from config per call (honors a live `/config paths.workspace`
        # change) and ensure it exists before writing.
        workspace = get_config().path("workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        target_path = (workspace / file_path).resolve()

        if not target_path.is_relative_to(workspace):
            return "Invalid file path: outside the workspace."
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
