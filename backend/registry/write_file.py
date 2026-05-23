from langchain.tools import tool
from pathlib import Path


WORKSPACE_DIR = Path("database/workspace").resolve()


# internal tools are written in house, external tools are implemented using integrastions
# need to sandbox to this to the workspace dir4ectory only
@tool
def write_file(file_path: str, content: str, overwrite: bool = False):
    """Writes the given content to a file at the specified file path. Takes the file path, content, and operation as input parameters. Should be a string The file path should be a string. The file path should be a valid path to a file."""
    target_path = (WORKSPACE_DIR / file_path).resolve()

    if not str(target_path).startswith(str(WORKSPACE_DIR)):
        return "Invalid file path."
    if overwrite:
        # overwrrite is true
        with open(target_path, "w") as file:
            file.write(content)
            return "File overwritten successfully"
    else:
        with open(target_path, "a") as file:
            file.write(content)
            return "Content appended to file successfully"
