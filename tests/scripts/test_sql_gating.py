import re
from pathlib import Path


def test_scripts_sql_include_wire_check_status():
    """Ensure SQL queries in scripts selecting labeled articles also check for a completed wire_check_status.

    This prevents premature export/backfills that include 'labeled' articles before wire checks complete.
    """
    root = Path("scripts")
    pattern_text = re.compile(
        r"(?:text|sql_text)\s*\(\s*([\"']{3})(.*?)(\1)\s*\)", re.DOTALL | re.IGNORECASE
    )
    pattern_status = re.compile(r"status\s*=\s*['\"]labeled['\"]", re.IGNORECASE)
    pattern_wire = re.compile(r"wire_check_status", re.IGNORECASE)

    whitelist_paths = [
        # Migration or admin scripts that intentionally set status or query unlabeled rows
        "scripts/migrate_labeled_articles.py",
        "scripts/demos/",
        "scripts/debug/",
    ]

    issues = []

    for f in root.rglob("*.py"):
        path = str(f)
        if any(w in path for w in whitelist_paths):
            continue
        text = f.read_text(encoding="utf8")
        for m in pattern_text.finditer(text):
            sql = m.group(2).lower()
            if "status" not in sql or "labeled" not in sql or "where" not in sql:
                continue
            if pattern_status.search(sql) and not pattern_wire.search(sql):
                issues.append(
                    (path, sql.strip()[:240].replace("\n", " "))
                )  # show a snippet

    assert (
        not issues
    ), f"Found SQL queries selecting labeled articles without wire_check_status: {issues}"
