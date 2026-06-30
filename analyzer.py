"""LLM-based semantic analysis using LangChain + Claude.

Files are grouped by Spring layer (controllers, services, repositories, models)
rather than chunked arbitrarily by token count. Keeping semantically related
files together gives Claude better context and maps to what the assignment asks:
purpose, method descriptions, and noteworthy aspects.
"""
import json
import logging
from typing import Dict, List, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-8"
# ~100K tokens per batch (4 chars ≈ 1 token); Claude's context is 1M but we stay conservative
MAX_CHARS_PER_BATCH = 400_000


def _truncate(text: str, max_chars: int = 50_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


def _batches(files: List[Dict], max_chars: int = MAX_CHARS_PER_BATCH) -> List[List[Dict]]:
    """Split a list of file dicts into batches that stay within the char budget."""
    batches, current, used = [], [], 0
    for f in files:
        size = len(f.get("content", ""))
        if current and used + size > max_chars:
            batches.append(current)
            current, used = [f], size
        else:
            current.append(f)
            used += size
    if current:
        batches.append(current)
    return batches


def _format_files(files: List[Dict]) -> str:
    parts = []
    for f in files:
        parts.append(f"### {f['class_name']} ({f['relative_path']})\n\n{_truncate(f['content'])}")
    return "\n\n---\n\n".join(parts)


class CodeAnalyzer:
    def __init__(self):
        self.llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=8192)
        self.parser = JsonOutputParser()

    # ------------------------------------------------------------------ public

    def analyze(self, code_data: Dict) -> Dict:
        files = code_data["files"]
        stats = code_data["stats"]

        layers = self._split_by_layer(files)

        logger.info("Extracting project overview...")
        overview = self._overview(files, stats)

        logger.info("Extracting key methods by layer...")
        methods = []
        for layer_name, layer_files in layers.items():
            if not layer_files:
                continue
            logger.info("  → %s (%d files)", layer_name, len(layer_files))
            for batch in tqdm(_batches(layer_files), desc=layer_name, unit="batch"):
                methods.extend(self._methods(batch, layer_name))

        return {"project_overview": overview, "key_methods": methods}

    # ------------------------------------------------------------------ layer grouping

    def _split_by_layer(self, files: List[Dict]) -> Dict[str, List[Dict]]:
        layers: Dict[str, List[Dict]] = {
            "controllers": [], "services": [], "repositories": [],
            "models": [], "config": [], "other": [],
        }
        for f in files:
            name = f["class_name"]
            anns = set(f.get("class_annotations", []))
            pkg = f.get("package", "")
            if "Controller" in name or "RestController" in anns or "controller" in pkg:
                layers["controllers"].append(f)
            elif "Service" in name or "Service" in anns or "service" in pkg:
                layers["services"].append(f)
            elif "Repository" in name or "Repo" in name or "Repository" in anns or "repository" in pkg:
                layers["repositories"].append(f)
            elif "Entity" in anns or "Embeddable" in anns or "model" in pkg or "entity" in pkg or "domain" in pkg:
                layers["models"].append(f)
            elif "Configuration" in anns or "config" in pkg:
                layers["config"].append(f)
            else:
                layers["other"].append(f)
        return layers

    # ------------------------------------------------------------------ LLM chains

    def _overview(self, files: List[Dict], stats: Dict) -> Dict:
        structure = "\n".join(f["relative_path"] for f in files)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a software architect. Respond with valid JSON only — no markdown fences."),
            ("human", (
                "Analyze this Spring Boot project and return a JSON object with keys:\n"
                "name, description, purpose, domain, architecture_pattern,\n"
                "technology_stack (list of {{name, purpose}}), key_features (list),\n"
                "api_design_principles (list), notable_aspects (list).\n\n"
                "Project stats: {stats}\n\nFile tree:\n{structure}"
            )),
        ])
        chain = prompt | self.llm | self.parser
        try:
            return chain.invoke({"stats": json.dumps(stats), "structure": structure})
        except Exception as e:
            logger.error("Overview extraction failed: %s", e)
            return {"error": str(e)}

    def _methods(self, files: List[Dict], layer: str) -> List[Dict]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior Java developer. Respond with valid JSON only — no markdown fences."),
            ("human", (
                "Extract key methods from these Spring Boot {layer} files.\n"
                "Return a JSON object with a single key 'methods', whose value is a list.\n"
                "Each item: class_name, package, method_name, full_signature, description,\n"
                "http_method (GET/POST/PUT/DELETE/PATCH or null), endpoint (path or null),\n"
                "parameters (list of strings), return_description, business_logic, annotations (list).\n\n"
                "Files:\n{files}"
            )),
        ])
        chain = prompt | self.llm | self.parser
        try:
            result = chain.invoke({"layer": layer, "files": _format_files(files)})
            return result.get("methods", []) if isinstance(result, dict) else []
        except Exception as e:
            logger.error("Method extraction failed for batch in %s: %s", layer, e)
            return []
