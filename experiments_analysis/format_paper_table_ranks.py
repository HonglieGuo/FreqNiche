"""Apply best/runner-up formatting to comparable manuscript result cells.

Explicit scopes prevent mixing datasets, evaluation tracks, or descriptive
quantities. No numeric value, uncertainty, label or method order is changed.
"""
from pathlib import Path
import re
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[1] / "Paper/FreqNiche-zh"


def content(cell):
    value = cell.strip()
    match = re.fullmatch(r"\\textbf\{(.*)\}", value)
    if match:
        value = match[1]
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1]
    match = re.fullmatch(r"\\(?:boldsymbol|mathbf|underline)\{(.*)\}", value)
    if match:
        value = match[1]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:\s*\\pm\s*\d+(?:\.\d+)?)?", value):
        raise ValueError(f"Not a numeric result cell: {cell}")
    return value


def rank(lines, columns, direction=1):
    cells = [line.split("&") for line in lines]
    for col in columns:
        bodies = [content(row[col].replace(r"\\", "").strip()) for row in cells]
        means = [Decimal(re.match(r"[+-]?\d+(?:\.\d+)?", body)[0]) for body in bodies]
        ordered = sorted(set(means), reverse=(direction == 1))
        for row, body, mean in zip(cells, bodies, means):
            marker = "boldsymbol" if mean == ordered[0] else "underline" if len(ordered) > 1 and mean == ordered[1] else None
            suffix = r"\\" if r"\\" in row[col] else ""
            row[col] = " $" + ("\\" + marker + "{" + body + "}" if marker else body) + "$" + suffix
    return ["&".join(row) for row in cells]


def process_tabular(block, label, panel):
    lines = block.splitlines()
    if label in {"tab:seaad", "tab:niche_benchmark", "tab:xenium_summary", "tab:refinement", "tab:contribution", "tab:table7_xenium_benchmark"}:
        if label == "tab:seaad":
            cols, down, descriptive = [2, 3], [], []
        elif label == "tab:niche_benchmark":
            cols, down, descriptive = [1, 2, 3, 4], [], []
        elif label == "tab:xenium_summary":
            cols, down, descriptive = [2, 3, 4], [], []
        elif label == "tab:refinement":
            cols, down, descriptive = [1, 3], [2], []
        elif label == "tab:contribution":
            cols, down, descriptive = [1], [], []
        else:
            cols, down, descriptive = [4, 6, 7, 8, 9], [], [5]
        indices = []
        for i, line in enumerate(lines):
            if "&" not in line:
                continue
            try:
                row = line.split("&")
                for col in cols + down:
                    content(row[col].replace(r"\\", "").strip())
            except (IndexError, ValueError):
                continue
            indices.append(i)
        selected = [lines[i] for i in indices]
        selected = rank(selected, cols)
        if down:
            selected = rank(selected, down, -1)
        # Absolute effective rank cannot be fairly ranked across dimensions.
        for j, line in enumerate(selected):
            cells = line.split("&")
            for col in descriptive:
                cells[col] = " $" + content(cells[col]) + "$ "
            selected[j] = "&".join(cells)
        for i, line in zip(indices, selected):
            lines[i] = line
    elif label in {"tab:skin_clusterer_sensitivity", "tab:skin_inductive_transductive"}:
        cols = [2, 3, 4, 5] if label == "tab:skin_clusterer_sensitivity" else [4, 5, 6]
        groups, current = [], []
        for i, line in enumerate(lines):
            if r"\midrule" in line and current:
                groups.append(current); current = []
            if line.lstrip().startswith("&") and re.search(r"\$(?:\\(?:boldsymbol|underline|mathbf)\{)?[+-]?\d+\.\d+", line):
                current.append(i)
        if current:
            groups.append(current)
        for indices in groups:
            # The inductive table stores numeric results on a second line.
            if label == "tab:skin_inductive_transductive":
                selected = ["dummy & dummy & dummy & dummy " + lines[i] for i in indices]
                replaced = rank(selected, cols)
                replaced = [line.split("&", 4)[-1] for line in replaced]
                replaced = ["&" + line for line in replaced]
            else:
                replaced = rank([lines[i] for i in indices], cols)
            for i, line in zip(indices, replaced):
                lines[i] = line
    elif label == "tab:multidataset_ablation":
        indices = [i for i, line in enumerate(lines) if "&" in line and r"\pm" in line]
        if panel == 0:
            selected = rank([lines[i] for i in indices], [1, 3, 4, 5, 6])
            selected = rank(selected, [2], -1)
            for i, line in zip(indices, selected):
                lines[i] = line
        else:
            for i in indices:
                cells = lines[i].split("&")
                row = " & ".join(["complete", cells[1], "", "", "", ""])
                other = " & ".join(["ablated", cells[2], "", "", "", ""])
                result = rank([row, other], [1], -1 if r"\downarrow" in cells[0] else 1)
                cells[1] = result[0].split("&")[1]
                cells[2] = result[1].split("&")[1]
                lines[i] = "&".join(cells)
    return "\n".join(lines)


def main():
    files = [ROOT / "sections/04_experiments_and_results.tex", ROOT / "sections/07_supplementary_methods.tex", *sorted((ROOT / "tables").glob("*.tex"))]
    for path in files:
        before = path.read_text(encoding="utf-8")
        def table(match):
            block = match[0]
            label = re.search(r"\\label\{([^}]+)\}", block)
            if not label:
                return block
            count = -1
            def tabular(match):
                nonlocal count
                count += 1
                return process_tabular(match[0], label[1], count)
            return re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", tabular, block, flags=re.S)
        after = re.sub(r"\\begin\{table\}.*?\\end\{table\}", table, before, flags=re.S)
        # A formatting operation must preserve the complete numeric sequence.
        assert re.findall(r"[+-]?\d+(?:\.\d+)?", before) == re.findall(r"[+-]?\d+(?:\.\d+)?", after), path
        if after != before:
            path.write_text(after, encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
