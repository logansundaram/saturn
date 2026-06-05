"""
Workspace file tools — read, write, and list files inside the sandboxed workspace.

Every path is resolved against `config.path("workspace")` *per call* (so a live
`/config paths.workspace` change is honored without a restart) and checked with `is_relative_to`
so a tool call can never escape the workspace. The sandbox is the boundary; `write_file` is the
one side-effecting tool here (it goes through the approval gate via registry.TOOL_RISK).
"""

import time

from langchain.tools import tool

from config import get_config
from stores.document_registry import register_workspace_file


@tool
def read_file(file_path: str):
    """Reads the contents of a file in the workspace and returns it as a string. file_path is relative to the workspace root."""
    start = time.perf_counter()
    try:
        workspace = get_config().path("workspace")
        target_path = (workspace / file_path).resolve()

        if not target_path.is_relative_to(workspace):
            return "Invalid file path: outside the workspace."
        with open(target_path, "r") as file:
            content = file.read()
        return content
    finally:
        print(f"read_file : {time.perf_counter() - start:.4f}s")


@tool
def write_file(file_path: str, content: str, overwrite: bool = True):
    """Writes content to a file in the workspace. file_path is relative to the workspace root. content is the text to write. overwrite=True (default) replaces the file's contents; pass overwrite=False to append to the existing file instead."""
    start = time.perf_counter()
    try:
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


@tool
def list_directory(directory: str = "."):
    """Lists the files and folders inside a workspace directory. directory is a path relative to the workspace root. Use '.' to list the workspace root."""
    start = time.perf_counter()
    try:
        workspace = get_config().path("workspace")
        target_path = (workspace / directory).resolve()

        if not target_path.is_relative_to(workspace):
            return "Invalid directory path: outside the workspace."

        if not target_path.is_dir():
            return "Path is not a directory."

        return [item.name for item in target_path.iterdir()]
    finally:
        print(f"list_directory : {time.perf_counter() - start:.4f}s")
