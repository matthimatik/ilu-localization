"""HTML report of the fused records: summary counts, table, overlay + scene links."""

import base64
import html
from pathlib import Path

from core.visualization.draw import record_table_cells
from schemas.association import Associations
from schemas.ilu import IdentificationMethod, ILURecords
from schemas.vision import TextSpots

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 72rem;
       color: #1c1c1c; padding: 0 1rem; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
.summary span { display: inline-block; background: #eef1f5; border-radius: 4px;
                padding: .2rem .6rem; margin: .2rem .4rem .2rem 0; }
img.overlay { max-width: 100%; border: 1px solid #ccc; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { border: 1px solid #ccc; padding: .35rem .6rem; text-align: left; }
th { background: #f4f6f8; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { background: #f4f6f8; padding: 0 .3rem; }
.links a { margin-right: 1rem; }
"""
# Columns rendered right-aligned (pose, ocr, text_match, #spots).
_NUM_COLUMNS = {6, 7, 8, 9}


def _escape(value: object) -> str:
    return html.escape(str(value))


def _summary_counts(records: ILURecords) -> list[str]:
    counts = len(records.records)
    by_method = {
        method: sum(1 for r in records.records if r.identification_method == method)
        for method in IdentificationMethod
    }
    by_category: dict[str, int] = {}
    for record in records.records:
        by_category[record.category] = by_category.get(record.category, 0) + 1
    lines = [f"{counts} records"]
    lines += [
        f"{value} {method.value}" for method, value in by_method.items() if value
    ]
    lines += [f"{value} {category}" for category, value in sorted(by_category.items())]
    return lines


def build_report(
    records: ILURecords,
    overlay_jpeg: bytes | None = None,
    links: dict[str, str] | None = None,
    associations: Associations | None = None,
    text_spots: TextSpots | None = None,
    title: str = "ILU localization result",
) -> str:
    """Self-contained HTML page (overlay inlined as base64)."""
    links = links or {}
    headers, cells_rows = record_table_cells(
        records.records, text_spots, associations
    )

    rows = []
    for cells in cells_rows:
        rendered = []
        for column, cell in enumerate(cells):
            if column == 0:
                rendered.append(f"<td><code>{_escape(cell)}</code></td>")
            elif column == 3 and cell != "—":
                rendered.append(f"<td><code>{_escape(cell)}</code></td>")
            elif column in _NUM_COLUMNS:
                rendered.append(f'<td class="num">{_escape(cell)}</td>')
            else:
                rendered.append(f"<td>{_escape(cell)}</td>")
        rows.append("<tr>" + "".join(rendered) + "</tr>")

    header = "<tr>" + "".join(f"<th>{_escape(h)}</th>" for h in headers) + "</tr>"
    summary = "".join(f"<span>{_escape(line)}</span>" for line in _summary_counts(records))
    overlay_html = ""
    if overlay_jpeg:
        encoded = base64.b64encode(overlay_jpeg).decode("ascii")
        overlay_html = (
            f'<img class="overlay" alt="overlay" '
            f'src="data:image/jpeg;base64,{encoded}">'
        )
    links_html = "".join(
        f'<a href="{_escape(href)}">{_escape(label)}</a>'
        for label, href in links.items()
    )
    links_html = f'<p class="links">{links_html}</p>' if links_html else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p class="summary">{summary}</p>
{links_html}
{overlay_html}
<h2>Records</h2>
<table>
{header}
{"".join(rows)}
</table>
</body>
</html>
"""


def write_report(html_text: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text)
    return path
