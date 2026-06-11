from __future__ import annotations

import re


NODE_DECL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9_]+)\[([^\]\n]+)\]")
SUBGRAPH_RE = re.compile(r"^(\s*)subgraph\s+(.+?)\s*$")


def sanitize_mermaid(source: str) -> str:
    lines = [line.rstrip() for line in strip_fence(source).splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].strip().startswith("classDiagram"):
        return ""

    id_map = collect_invalid_node_ids(lines)
    sanitized = [sanitize_mermaid_line(line, id_map) for line in lines]
    return "\n".join(sanitized)


def strip_fence(source: str) -> str:
    text = source.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:mermaid)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def collect_invalid_node_ids(lines: list[str]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for line in lines:
        for match in NODE_DECL_RE.finditer(line):
            node_id = match.group(1)
            if not is_valid_mermaid_id(node_id):
                ids[node_id] = sanitize_id(node_id)
    return ids


def sanitize_mermaid_line(line: str, id_map: dict[str, str]) -> str:
    line = line.replace("\t", "  ")
    line = sanitize_subgraph(line)
    line = NODE_DECL_RE.sub(lambda match: quote_node_label(match, id_map), line)

    for old, new in id_map.items():
        line = replace_outside_quotes(line, old, new)

    return line


def sanitize_subgraph(line: str) -> str:
    match = SUBGRAPH_RE.match(line)
    if not match:
        return line

    indent, raw_name = match.groups()
    if "[" in raw_name or raw_name.lower().startswith(("end", "direction ")):
        return line

    title = raw_name.strip()
    graph_id = sanitize_id(title)
    if not graph_id:
        return line
    return f"{indent}subgraph {graph_id} [{title}]"


def quote_node_label(match: re.Match[str], id_map: dict[str, str]) -> str:
    node_id = id_map.get(match.group(1), match.group(1))
    label = match.group(2).strip()
    label = label.replace('"', "'")
    return f'{node_id}["{label}"]'


def is_valid_mermaid_id(node_id: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", node_id))


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not cleaned:
        return ""
    if cleaned[0].isdigit():
        cleaned = f"N_{cleaned}"
    return cleaned


def replace_outside_quotes(line: str, old: str, new: str) -> str:
    parts = re.split(r'("[^"]*")', line)
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])"
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(pattern, new, parts[index])
    return "".join(parts)
