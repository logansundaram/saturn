from pathlib import Path
import webbrowser

# might need rename this file as utilities or smthing like that
# contain useful info to visualize debug and benchmark the agents

GRAPH_DIR = Path("logging/graphs")


def print_graph(graph, name: str = "agent_graph", open_browser: bool = True):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    mermaid_path = GRAPH_DIR / f"{name}.mmd"
    png_path = GRAPH_DIR / f"{name}.png"

    drawable = graph.get_graph()

    # Save Mermaid source
    mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")

    # Save PNG image
    png_path.write_bytes(drawable.draw_mermaid_png())

    print(f"Graph Mermaid saved to: {mermaid_path.resolve()}")
    print(f"Graph PNG saved to: {png_path.resolve()}")

    if open_browser:
        webbrowser.open(png_path.resolve().as_uri())
