"""stores/rag.py loaders — the per-format ingest paths (html/csv/docx joined txt/md/pdf) and the
'what counts as a document' rules. No embedding happens here: only the load/normalize layer."""

import pytest

from stores.rag import (
    SUPPORTED_EXTENSIONS,
    _csv_to_text,
    _html_to_text,
    _load_file_docs,
    ingest_file,
    iter_documents,
)


def test_ingest_refuses_basename_clobber(isolated_paths, tmp_path):
    """Two different source files sharing a basename must not silently overwrite each other in
    the corpus — the first document would be permanently gone with no warning."""
    corpus = isolated_paths / "database" / "documents"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "report.md").write_text("project A report", encoding="utf-8")

    other = tmp_path / "elsewhere"
    other.mkdir()
    clash = other / "report.md"
    clash.write_text("project B report — different content", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ingest_file(str(clash))
    # The original document is untouched.
    assert (corpus / "report.md").read_text(encoding="utf-8") == "project A report"


def test_supported_extensions_cover_new_formats():
    for ext in (".txt", ".md", ".pdf", ".html", ".htm", ".csv", ".docx"):
        assert ext in SUPPORTED_EXTENSIONS


def test_docs_help_advertises_every_supported_format():
    """/docs --help under-reported the formats as (pdf/txt/md) and the shipped html/csv/docx
    loaders went undiscovered. The details string stays a LITERAL on purpose — knowledge.py
    deliberately lazy-imports stores.rag inside handlers, and building the help dynamically
    would drag the loader stack into command registration at startup — so the drift guard
    lives here instead: every supported suffix must appear in the registered help text."""
    import commands  # noqa: F401 — two-phase registration of every built-in module
    from commands._framework import COMMANDS

    details = COMMANDS["docs"].details.lower()
    for ext in SUPPORTED_EXTENSIONS:
        assert ext.lstrip(".") in details, f"/docs --help does not mention {ext}"


def test_html_to_text_strips_markup_and_script():
    raw = (
        "<html><head><style>p{color:red}</style></head><body>"
        "<h1>Title</h1><p>Hello <b>world</b></p>"
        "<script>var evil = 'payload';</script></body></html>"
    )
    out = _html_to_text(raw)
    assert "Hello" in out and "world" in out
    assert "<" not in out
    assert "evil" not in out and "color:red" not in out


def test_html_entities_unescaped():
    out = _html_to_text("<p>a &amp; b &lt;ok&gt;</p>")
    assert "a & b" in out


def test_csv_gets_schema_header():
    raw = "name,age,city\nada,36,london\n"
    out = _csv_to_text(raw)
    assert out.startswith("[columns: name, age, city]")
    assert "ada,36,london" in out  # rows stay verbatim


def test_csv_without_parseable_header_passes_through():
    assert _csv_to_text("") == ""


def test_load_file_docs_html_and_csv(isolated_paths):
    docs_dir = isolated_paths / "database" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "page.html").write_text("<p>html body text</p>", encoding="utf-8")
    (docs_dir / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    source, docs, full_text = _load_file_docs(docs_dir / "page.html")
    assert source == "page.html"
    assert docs[0].metadata["source"] == "page.html"
    assert "html body text" in full_text and "<p>" not in full_text

    source, docs, full_text = _load_file_docs(docs_dir / "data.csv")
    assert full_text.startswith("[columns: a, b]")


def test_load_file_docs_docx(isolated_paths):
    docx = pytest.importorskip("docx", reason="python-docx not installed")
    docs_dir = isolated_paths / "database" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    d = docx.Document()
    d.add_paragraph("First paragraph of prose.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "cell-a"
    table.rows[0].cells[1].text = "cell-b"
    d.save(str(docs_dir / "doc.docx"))

    source, docs, full_text = _load_file_docs(docs_dir / "doc.docx")
    assert "First paragraph of prose." in full_text
    assert "cell-a\tcell-b" in full_text


def test_iter_documents_extension_casing_and_dotfiles(isolated_paths):
    docs_dir = isolated_paths / "database" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "NOTES.MD").write_text("upper-ext", encoding="utf-8")
    (docs_dir / ".manifest.md").write_text("hidden", encoding="utf-8")
    (docs_dir / "skip.xyz").write_text("unsupported", encoding="utf-8")
    names = {p.name for p in iter_documents()}
    assert names == {"NOTES.MD"}
