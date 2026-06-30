"""Static analysis of Java source files using regex (no AST parser needed)."""
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PACKAGE_RE = re.compile(r"^package\s+([\w.]+)\s*;", re.MULTILINE)
_IMPORT_RE = re.compile(r"^import\s+(?:static\s+)?([\w.*]+)\s*;", re.MULTILINE)
_CLASS_RE = re.compile(
    r"(?:public|protected|private)?\s*(?:abstract\s+|final\s+)*"
    r"(class|interface|enum|@interface)\s+(\w+)",
    re.MULTILINE,
)
_ANNOTATION_RE = re.compile(r"@(\w+)")
# keywords that each add +1 to cyclomatic complexity
_CC_RE = re.compile(r"\b(if|else if|for|while|do|case|catch|finally)\b|&&|\|\||\?(?!\?)")

_MODIFIERS = frozenset({
    "public", "protected", "private", "static", "final",
    "synchronized", "abstract", "native", "default", "strictfp",
})
_SKIP_NAMES = frozenset({"if", "for", "while", "switch", "catch", "new"})


@dataclass
class MethodInfo:
    name: str
    signature: str
    annotations: List[str]
    line_number: int
    cyclomatic_complexity: int
    parameter_count: int


@dataclass
class FileInfo:
    relative_path: str
    package: str
    class_name: str
    class_type: str
    class_annotations: List[str]
    imports: List[str]
    methods: List[MethodInfo]
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    content: str  # raw source kept for LLM step; stripped before JSON output


class CodeProcessor:
    def __init__(self, repo_dir: str):
        self.repo_dir = Path(repo_dir)

    def process(self) -> Dict:
        files = self._discover()
        logger.info("Found %d Java source files", len(files))
        infos = [r for p in files if (r := self._analyse(p)) is not None]
        return self._compile(infos)

    # ------------------------------------------------------------------ discovery

    def _discover(self) -> List[Path]:
        src = self.repo_dir / "src" / "main" / "java"
        if src.exists():
            return sorted(src.rglob("*.java"))
        return sorted(
            p for p in self.repo_dir.rglob("*.java")
            if "test" not in p.parts and "Test" not in p.name
        )

    # ------------------------------------------------------------------ per-file

    def _analyse(self, path: Path) -> Optional[FileInfo]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Cannot read %s: %s", path, e)
            return None

        class_name, class_type, class_ann = self._class_info(content)
        if not class_name:
            return None

        lines = content.splitlines()
        lc = self._line_counts(lines)
        return FileInfo(
            relative_path=str(path.relative_to(self.repo_dir)),
            package=self._package(content),
            class_name=class_name,
            class_type=class_type,
            class_annotations=class_ann,
            imports=_IMPORT_RE.findall(content),
            methods=self._methods(lines),
            total_lines=lc["total"],
            code_lines=lc["code"],
            comment_lines=lc["comment"],
            blank_lines=lc["blank"],
            content=content,
        )

    def _package(self, src: str) -> str:
        m = _PACKAGE_RE.search(src)
        return m.group(1) if m else ""

    def _class_info(self, src: str) -> Tuple[str, str, List[str]]:
        m = _CLASS_RE.search(src)
        if not m:
            return "", "class", []
        annotations = []
        for line in reversed(src[: m.start()].splitlines()):
            s = line.strip()
            if s.startswith("@"):
                annotations.extend(_ANNOTATION_RE.findall(s))
            elif s and not s.startswith(("/", "*")):
                break
        return m.group(2), m.group(1), annotations

    def _methods(self, lines: List[str]) -> List[MethodInfo]:
        results, pending_ann = [], []
        for idx, line in enumerate(lines):
            s = line.strip()
            if s.startswith("@"):
                pending_ann.extend(_ANNOTATION_RE.findall(s))
                continue
            if not s:
                pending_ann.clear()
                continue
            if s.startswith(("/", "*")):
                continue
            if self._is_method(s):
                name = self._method_name(s)
                if name and name not in _SKIP_NAMES:
                    body = "\n".join(lines[idx: idx + 150])
                    results.append(MethodInfo(
                        name=name,
                        signature=s.rstrip("{").strip(),
                        annotations=list(pending_ann),
                        line_number=idx + 1,
                        cyclomatic_complexity=1 + len(_CC_RE.findall(body)),
                        parameter_count=self._param_count(s),
                    ))
            pending_ann.clear()
        return results

    def _is_method(self, s: str) -> bool:
        return (
            "(" in s
            and bool(set(s.split()) & _MODIFIERS)
            and not (";'" in s and "{" not in s)
        )

    def _method_name(self, s: str) -> Optional[str]:
        parts = s[: s.find("(")].split()
        return parts[-1] if parts else None

    def _param_count(self, s: str) -> int:
        m = re.search(r"\(([^)]*)\)", s)
        return len(m.group(1).split(",")) if m and m.group(1).strip() else 0

    def _line_counts(self, lines: List[str]) -> Dict[str, int]:
        total, blank, comment, in_block = len(lines), 0, 0, False
        for line in lines:
            s = line.strip()
            if in_block:
                comment += 1
                if "*/" in s:
                    in_block = False
            elif not s:
                blank += 1
            elif s.startswith("//"):
                comment += 1
            elif s.startswith("/*"):
                comment += 1
                if "*/" not in s[2:]:
                    in_block = True
        return {"total": total, "blank": blank, "comment": comment, "code": max(0, total - blank - comment)}

    # ------------------------------------------------------------------ compile

    def _compile(self, infos: List[FileInfo]) -> Dict:
        all_cc = [m.cyclomatic_complexity for fi in infos for m in fi.methods]
        avg_cc = round(sum(all_cc) / len(all_cc), 2) if all_cc else 0.0
        top5 = sorted(infos, key=lambda fi: max((m.cyclomatic_complexity for m in fi.methods), default=0), reverse=True)[:5]

        return {
            "files": [
                {
                    "relative_path": fi.relative_path,
                    "package": fi.package,
                    "class_name": fi.class_name,
                    "class_type": fi.class_type,
                    "class_annotations": fi.class_annotations,
                    "imports": fi.imports,
                    "methods": [
                        {
                            "name": m.name,
                            "signature": m.signature,
                            "annotations": m.annotations,
                            "line_number": m.line_number,
                            "cyclomatic_complexity": m.cyclomatic_complexity,
                            "parameter_count": m.parameter_count,
                        }
                        for m in fi.methods
                    ],
                    "metrics": {
                        "total_lines": fi.total_lines,
                        "code_lines": fi.code_lines,
                        "comment_lines": fi.comment_lines,
                        "blank_lines": fi.blank_lines,
                        "method_count": len(fi.methods),
                        "import_count": len(fi.imports),
                    },
                    "content": fi.content,  # stripped by formatter before JSON output
                }
                for fi in infos
            ],
            "stats": {
                "total_files": len(infos),
                "total_lines": sum(fi.total_lines for fi in infos),
                "total_code_lines": sum(fi.code_lines for fi in infos),
                "total_methods": sum(len(fi.methods) for fi in infos),
                "avg_cyclomatic_complexity": avg_cc,
                "most_complex_files": [fi.relative_path for fi in top5],
                "packages": sorted({fi.package for fi in infos if fi.package}),
            },
        }
