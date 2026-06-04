from langchain.tools import tool

from config import get_config


@tool
def list_directory(directory: str = "."):
    """Lists the files and folders inside a workspace directory. directory is a path relative to the workspace root. Use '.' to list the workspace root."""
    # Resolve the workspace from config per call so a live `/config paths.workspace` change
    # is honored without a restart.
    workspace = get_config().path("workspace")
    target_path = (workspace / directory).resolve()

    if not target_path.is_relative_to(workspace):
        return "Invalid directory path: outside the workspace."
    if not target_path.is_dir():
        return "Path is not a directory."

    return [item.name for item in target_path.iterdir()]
