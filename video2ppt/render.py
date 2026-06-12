from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .io_utils import read_json
from .mermaid import sanitize_mermaid
from .notes import format_timestamp


def render_html(notes_path: Path, validation_path: Path | None, output_path: Path) -> None:
    notes = read_json(notes_path)
    validation = read_json(validation_path) if validation_path and validation_path.exists() else {"results": []}
    validation_by_claim = {
        item["claim"]: item
        for item in validation.get("results", [])
        if isinstance(item, dict) and item.get("claim")
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(notes, validation, validation_by_claim), encoding="utf-8")


def build_html(notes: dict[str, Any], validation: dict[str, Any], validation_by_claim: dict[str, Any]) -> str:
    title = html_text(notes.get("title", "Lecture Notes"))
    source_video = html_text(notes.get("source_video", ""))
    summary = html_text(validation.get("summary", ""))
    nav = "\n".join(
        f'<a href="#section-{index}">{index}. {html_text(section.get("title", "Untitled"))}</a>'
        for index, section in enumerate(notes.get("sections", []), start=1)
    )
    sections = "\n".join(
        build_section(index, section, validation_by_claim)
        for index, section in enumerate(notes.get("sections", []), start=1)
    )

    credibility = f'<p class="summary">Credibility: {summary}</p>' if summary else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
      <style>{CSS}</style>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: false, theme: "default", securityLevel: "strict" }});
    for (const [index, element] of [...document.querySelectorAll(".mermaid-source")].entries()) {{
      const source = element.textContent.trim();
      try {{
        const rendered = await mermaid.render(`mermaid-diagram-${{index}}`, source);
        if (rendered.svg.includes("Syntax error in text") || rendered.svg.includes("mermaid version")) {{
          throw new Error("Mermaid returned an error SVG");
        }}
        const container = document.createElement("div");
        container.className = "mermaid-rendered";
        container.innerHTML = rendered.svg;
        if (container.textContent.includes("Syntax error in text") || container.textContent.includes("mermaid version")) {{
          throw new Error("Mermaid rendered an error diagram");
        }}
        element.replaceWith(container);
      }} catch (error) {{
        console.warn("Mermaid diagram could not be rendered", error);
        element.classList.add("mermaid-fallback");
        element.insertAdjacentHTML("beforebegin", '<p class="aid-note">Diagram source shown because Mermaid could not render this block.</p>');
      }}
    }}
  </script>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="index-title">{title}</div>
      <nav>{nav}</nav>
    </aside>
    <main>
      <h1>{title}</h1>
      <p class="source">Source: {source_video}</p>
      {credibility}
      {sections}
    </main>
  </div>
</body>
</html>
"""


def build_section(index: int, section: dict[str, Any], validation_by_claim: dict[str, Any]) -> str:
    title = html_text(section.get("title", "Untitled"))
    timestamps = "".join(
        f'<span class="timestamp">{format_timestamp(float(stamp))}</span>'
        for stamp in section.get("timestamps", [])
        if is_number(stamp)
    )
    timestamp_html = f"<p>{timestamps}</p>" if timestamps else ""
    summary = f"<p>{html_text(section.get('summary', ''))}</p>" if section.get("summary") else ""
    bullets = build_list("Key Points", section.get("bullets", []))
    terms = build_terms(section.get("key_terms", []))
    examples = build_list("Examples", section.get("examples", []))
    aids = build_aids(section.get("aids", {}))
    claims = build_claims(section.get("claims", []), validation_by_claim)
    return f"""
<section class="slide" id="section-{index}">
  <h2>{title}</h2>
  {timestamp_html}
  {summary}
  {bullets}
  {terms}
  {examples}
  {aids}
  {claims}
</section>
"""


def build_list(title: str, items: list[Any]) -> str:
    cleaned = [html_text(item) for item in items if str(item).strip()]
    if not cleaned:
        return ""
    lis = "".join(f"<li>{item}</li>" for item in cleaned)
    return f"<h3>{html_text(title)}</h3><ul>{lis}</ul>"


def build_terms(items: list[Any]) -> str:
    cards: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("term", "")).strip():
            continue
        term = html_text(item.get("term", ""))
        definition = html_text(item.get("definition", ""))
        cards.append(f'<div class="term"><strong>{term}</strong><span>{definition}</span></div>')
    if not cards:
        return ""
    return f'<h3>Key Terms</h3><div class="term-grid">{"".join(cards)}</div>'


def build_claims(claims: list[Any], validation_by_claim: dict[str, Any]) -> str:
    blocks: list[str] = []
    for claim_value in claims:
        claim = str(claim_value).strip()
        if not claim:
            continue
        check = validation_by_claim.get(claim, {})
        status = str(check.get("status", "needs_review")).strip() or "needs_review"
        rationale = str(check.get("rationale", "")).strip()
        rationale_html = f'<div class="rationale">{html_text(rationale)}</div>' if rationale else ""
        sources = build_source_links(check.get("source_urls", []))
        blocks.append(
            '<div class="claim">'
            f"{html_text(claim)} "
            f'<span class="badge {html_attr(status)}">{html_text(status.replace("_", " "))}</span>'
            f"{rationale_html}{sources}</div>"
        )
    if not blocks:
        return ""
    return f"<h3>Credibility Checks</h3>{''.join(blocks)}"


def build_source_links(urls: Any) -> str:
    if not isinstance(urls, list):
        return ""
    links = []
    for index, url in enumerate(urls[:5], start=1):
        text = str(url).strip()
        if text:
            links.append(f'<a href="{html_attr(text)}" target="_blank" rel="noreferrer">source {index}</a>')
    if not links:
        return ""
    return f'<div class="sources">{" ".join(links)}</div>'


def build_aids(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""

    blocks = []
    blocks.extend(build_diagram_group("Architecture Diagrams", raw.get("architecture_diagrams", [])))
    blocks.extend(build_diagram_group("Flow Diagrams", raw.get("flow_diagrams", [])))
    blocks.extend(build_code_group("Code Snippets", raw.get("code_snippets", [])))
    blocks.extend(build_code_group("Syntax", raw.get("syntax", [])))
    blocks.extend(build_table_group(raw.get("tables", [])))
    if not blocks:
        return ""
    return "<h3>Learning Aids</h3>" + "".join(blocks)


def build_diagram_group(title: str, items: list[Any]) -> list[str]:
    blocks = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("mermaid", "")).strip():
            continue
        blocks.append(
            '<div class="aid">'
            f'<div class="aid-title">{html_text(item.get("title", title))}</div>'
            f'<pre class="mermaid-source">{html_text(sanitize_mermaid(str(item.get("mermaid", ""))))}</pre>'
            "</div>"
        )
    return blocks


def build_code_group(title: str, items: list[Any]) -> list[str]:
    blocks = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("code", "")).strip():
            continue
        language = html_text(item.get("language", "text"))
        explanation = str(item.get("explanation", "")).strip()
        explanation_html = f'<p class="aid-note">{html_text(explanation)}</p>' if explanation else ""
        blocks.append(
            '<div class="aid">'
            f'<div class="aid-title">{html_text(item.get("title", title))}</div>'
            f'<pre class="code-block"><code data-language="{language}">{html_text(item.get("code", ""))}</code></pre>'
            f"{explanation_html}</div>"
        )
    return blocks


def build_table_group(items: list[Any]) -> list[str]:
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        headers = [html_text(header) for header in item.get("headers", []) if str(header).strip()]
        rows = item.get("rows", [])
        if not headers or not rows:
            continue
        head = "".join(f"<th>{header}</th>" for header in headers)
        body_rows = []
        for row in rows:
            if not isinstance(row, list):
                continue
            cells = "".join(f"<td>{html_text(cell)}</td>" for cell in row)
            body_rows.append(f"<tr>{cells}</tr>")
        if body_rows:
            blocks.append(
                '<div class="aid">'
                f'<div class="aid-title">{html_text(item.get("title", "Table"))}</div>'
                f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
                "</div>"
            )
    return blocks


def html_text(value: Any) -> str:
    return escape(str(value), quote=False)


def html_attr(value: Any) -> str:
    return escape(str(value), quote=True)


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


CSS = """
:root {
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #697586;
  --line: #d9e0e8;
  --accent: #2563eb;
  --verified: #0f766e;
  --likely: #6d6a00;
  --review: #9a3412;
  --unsupported: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: 100vh;
}
aside {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
  border-right: 1px solid var(--line);
  background: #eef2f6;
  padding: 22px 18px;
}
main {
  padding: 34px;
  max-width: 1180px;
}
h1 {
  margin: 0 0 8px;
  font-size: 34px;
  line-height: 1.15;
  letter-spacing: 0;
}
h2 {
  margin: 0 0 12px;
  font-size: 24px;
  letter-spacing: 0;
}
h3 {
  margin: 22px 0 8px;
  font-size: 15px;
  text-transform: uppercase;
  letter-spacing: 0;
  color: var(--muted);
}
.source, .summary {
  color: var(--muted);
  margin: 0 0 24px;
}
.index-title {
  font-weight: 700;
  margin-bottom: 16px;
}
nav a {
  display: block;
  color: var(--ink);
  text-decoration: none;
  padding: 8px 10px;
  border-radius: 6px;
  margin: 2px 0;
  font-size: 14px;
}
nav a:hover {
  background: #dfe7f0;
  color: var(--accent);
}
.slide {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 28px;
  margin: 22px 0;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}
ul {
  padding-left: 22px;
  margin: 8px 0 0;
}
li {
  margin: 7px 0;
}
.term-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}
.term {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.term strong {
  display: block;
  margin-bottom: 3px;
}
.timestamp {
  display: inline-block;
  color: var(--muted);
  font-size: 13px;
  margin-right: 8px;
}
.claim {
  border-left: 4px solid var(--line);
  padding: 10px 12px;
  margin: 10px 0;
  background: #fbfcfe;
}
.aid {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px;
  margin: 12px 0;
  background: #fbfcfe;
}
.aid-title {
  font-weight: 700;
  margin-bottom: 8px;
}
.aid-note {
  color: var(--muted);
  margin: 8px 0 0;
  font-size: 14px;
}
.code-block {
  margin: 0;
  overflow: auto;
  border-radius: 6px;
  padding: 12px;
  background: #111827;
  color: #f9fafb;
  font-size: 13px;
  line-height: 1.45;
}
.mermaid-source {
  overflow: auto;
  margin: 0;
  padding: 12px;
  border-radius: 6px;
  background: #ffffff;
}
.mermaid-rendered {
  overflow: auto;
  padding: 8px;
  border-radius: 6px;
  background: #ffffff;
}
.mermaid-fallback {
  border: 1px dashed var(--review);
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #eef2f6;
}
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  margin-left: 6px;
  font-size: 12px;
  font-weight: 700;
  color: white;
}
.verified { background: var(--verified); }
.likely { background: var(--likely); }
.needs_review { background: var(--review); }
.unsupported { background: var(--unsupported); }
.rationale {
  color: var(--muted);
  margin-top: 4px;
  font-size: 14px;
}
.sources {
  margin-top: 6px;
  font-size: 13px;
}
.sources a {
  color: var(--accent);
  margin-right: 10px;
}
@media (max-width: 820px) {
  .layout { display: block; }
  aside {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  main { padding: 20px; }
  h1 { font-size: 28px; }
}
@media print {
  aside { display: none; }
  .layout { display: block; }
  main { padding: 0; }
  .slide { break-inside: avoid; box-shadow: none; }
}
"""
