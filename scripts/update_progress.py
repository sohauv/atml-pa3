"""Compute progress from verified mandatory checklist lines, without guessing."""
import re
from pathlib import Path


def main():
    path = Path("PROGRESS.md")
    text = path.read_text(encoding="utf-8")
    items = re.findall(r"^- \[([ x])\] ((?:G|T1|T2|T3)\d+)\b", text, flags=re.MULTILINE)
    if not items or len({key for _, key in items}) != len(items):
        raise ValueError("Missing or duplicate requirement IDs")
    complete = sum(state == "x" for state, _ in items)
    percent = 100 * complete / len(items)
    summary = (f"<!-- SUMMARY:START -->\n**Overall verified completion: {complete}/{len(items)} "
               f"mandatory requirements ({percent:.1f}%).**\n<!-- SUMMARY:END -->")
    text = re.sub(r"<!-- SUMMARY:START -->.*?<!-- SUMMARY:END -->", summary, text, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")
    print(f"{complete}/{len(items)} = {percent:.1f}% verified")


if __name__ == "__main__":
    main()
