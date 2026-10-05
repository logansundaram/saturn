"""Direct reads of the documents people actually have (pivot #2, 2026-09-29).

`read_file` returns a PDF / .docx / .xlsx as its text through core/doctext — "summarize the PDF
on my desktop" no longer depends on the knowledge base or the embedder — and refuses any other
binary file by name. `@file` attachments take the same path. The fixtures are built by hand
(a minimal PDF, a SpreadsheetML zip) so no writer library is needed. Offline.
"""

import zipfile

import pytest

from core import doctext, mentions, workspace
from tools.files import read_file
from tools.toolspec import ToolError


def _pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF: one Helvetica text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Contents {len(objs)} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


_NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _xlsx(path, sheets: dict[str, str], shared: list[str]):
    """Write an .xlsx with the given sheet name → <sheetData> body and shared strings."""
    with zipfile.ZipFile(path, "w") as zf:
        entries = "".join(f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>'
                          for i, n in enumerate(sheets, 1))
        zf.writestr("xl/workbook.xml", f"<workbook {_NS} {_R}><sheets>{entries}</sheets></workbook>")
        rels = "".join(
            f'<Relationship Id="rId{i}" Type="worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1))
        zf.writestr("xl/_rels/workbook.xml.rels",
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                    f'relationships">{rels}</Relationships>')
        sis = "".join(f"<si><t>{s}</t></si>" for s in shared)
        zf.writestr("xl/sharedStrings.xml", f"<sst {_NS}>{sis}</sst>")
        for i, body in enumerate(sheets.values(), 1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml",
                        f"<worksheet {_NS}><sheetData>{body}</sheetData></worksheet>")


@pytest.fixture
def folder(tmp_path, isolated_paths):
    launch = tmp_path / "launch"
    launch.mkdir()  # set_root falls back to HOME for a folder that doesn't exist
    assert workspace.set_root(launch) == launch.resolve()
    return launch


def test_read_file_returns_a_pdf_as_text_page_by_page(folder):
    (folder / "lease.pdf").write_bytes(_pdf(["Lease ends March 31", "Deposit 1200 dollars"]))
    out = read_file.invoke({"file_path": "lease.pdf"})
    assert "--- page 1 ---\nLease ends March 31" in out
    assert "--- page 2 ---\nDeposit 1200 dollars" in out


def test_a_pdf_without_a_text_layer_says_so(folder):
    (folder / "scan.pdf").write_bytes(_pdf(["", ""]))
    out = read_file.invoke({"file_path": "scan.pdf"})
    assert "no extractable text" in out and "2 page(s)" in out


def test_read_file_returns_a_docx_as_text(folder):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Dear Petra, Thursday works.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Room"
    table.rows[0].cells[1].text = "4B"
    d.save(str(folder / "letter.docx"))
    out = read_file.invoke({"file_path": "letter.docx"})
    assert "Dear Petra, Thursday works." in out and "Room\t4B" in out


def test_read_file_returns_each_xlsx_sheet_as_csv(folder):
    _xlsx(folder / "budget.xlsx", {
        "Rent": ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                 '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>1450.5</v></c></row>'),
        "Notes": '<row r="1"><c r="B1" t="inlineStr"><is><t>paid, on time</t></is></c>'
                 '<c r="C1" t="b"><v>1</v></c></row>',
        "Empty": "",
    }, shared=["Month", "Amount", "March"])
    out = read_file.invoke({"file_path": "budget.xlsx"})
    assert out == ("## Sheet: Rent\nMonth,Amount\nMarch,,1450.5\n\n"
                   '## Sheet: Notes\n,"paid, on time",TRUE')


def test_xlsx_rows_are_capped(folder, monkeypatch):
    monkeypatch.setattr(doctext, "_XLSX_MAX_ROWS", 2)
    rows = "".join(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(1, 6))
    _xlsx(folder / "big.xlsx", {"S": rows}, shared=[])
    assert read_file.invoke({"file_path": "big.xlsx"}) == "## Sheet: S\n1\n2\n… stopped at 2 rows"


def test_other_binary_files_are_refused_by_name(folder):
    (folder / "photo.jpg").write_bytes(b"\xff\xd8\xff\x00\x10JFIF")
    with pytest.raises(ToolError) as info:
        read_file.invoke({"file_path": "photo.jpg"})
    assert str(info.value).startswith("photo.jpg is a binary file (.jpg)")
    assert "PDF, .docx and .xlsx" in str(info.value)


def test_text_files_are_unchanged(folder):
    (folder / "page.html").write_text("<p>hi</p>", encoding="utf-8")
    assert read_file.invoke({"file_path": "page.html"}) == "<p>hi</p>"


def test_a_corrupt_document_raises_like_any_failed_read(folder):
    (folder / "broken.xlsx").write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        read_file.invoke({"file_path": "broken.xlsx"})


def test_an_attached_pdf_is_its_text(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(_pdf(["Quarterly total 88"]))
    block, paths = mentions.expand(f"summarize @{f}")
    assert paths == [str(f)] and "Quarterly total 88" in block


def test_an_unreadable_attached_document_is_an_inline_note(tmp_path):
    f = tmp_path / "broken.docx"
    f.write_bytes(b"not a zip")
    block, _ = mentions.expand(f"read @{f}")
    assert "[could not read" in block


def test_pdf_parser_warnings_never_reach_the_terminal(tmp_path):
    """pypdf logs a warning per odd object in a real-world PDF; with no handler configured
    those print to stderr, on top of the live TUI. Reading a PDF turns them down to errors."""
    import logging

    from core import doctext

    logging.getLogger("pypdf").setLevel(logging.NOTSET)
    try:
        doctext.pdf_pages(tmp_path / "missing.pdf")
    except Exception:
        pass
    assert logging.getLogger("pypdf").level == logging.ERROR
