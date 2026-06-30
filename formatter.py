"""Compile static metrics and LLM analysis into the final JSON structure.

Static metrics (LOC, complexity) and LLM-derived descriptions are kept in
separate top-level sections so consumers can trust each independently.
"""
from datetime import datetime, timezone
from typing import Dict, List

REPO_URL = "https://github.com/codejsha/spring-rest-sakila"
TOOL_VERSION = "1.0.0"
MODEL = "claude-opus-4-8"


def build_output(code_data: Dict, analysis: Dict) -> Dict:
    return {
        "metadata": {
            "project_url": REPO_URL,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "analyzer_version": TOOL_VERSION,
            "llm_model": MODEL,
        },
        "project_overview": analysis.get("project_overview", {}),
        "key_methods": analysis.get("key_methods", []),
        "complexity_metrics": _complexity(code_data),
    }


def _complexity(code_data: Dict) -> Dict:
    stats = code_data["stats"]
    per_file = [
        {
            "file": f["relative_path"],
            "class_name": f["class_name"],
            "class_type": f["class_type"],
            "package": f["package"],
            **f["metrics"],
            "max_method_complexity": max(
                (m["cyclomatic_complexity"] for m in f["methods"]), default=0
            ),
            "avg_method_complexity": round(
                sum(m["cyclomatic_complexity"] for m in f["methods"]) / len(f["methods"]), 2
            ) if f["methods"] else 0.0,
        }
        for f in code_data["files"]
    ]
    per_file.sort(key=lambda x: x["max_method_complexity"], reverse=True)

    return {
        "summary": {
            "total_files": stats["total_files"],
            "total_lines": stats["total_lines"],
            "total_code_lines": stats["total_code_lines"],
            "total_methods": stats["total_methods"],
            "avg_cyclomatic_complexity": stats["avg_cyclomatic_complexity"],
            "packages": stats["packages"],
            "most_complex_files": stats["most_complex_files"],
        },
        "per_file": per_file,
    }
