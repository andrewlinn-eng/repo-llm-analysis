"""Entry point for the codebase analysis tool."""
import argparse
import json
import logging
import os
from pathlib import Path

from fetcher import RepoFetcher
from processor import CodeProcessor
from analyzer import CodeAnalyzer
from formatter import build_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/codejsha/spring-rest-sakila"
CLONE_DIR = "./repos/spring-rest-sakila"
OUTPUT_FILE = "knowledge_output.json"


def main():
    parser = argparse.ArgumentParser(description="Codebase analysis via LLM")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--clone-dir", default=CLONE_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning if the repo directory already exists")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set")

    # 1. Fetch
    if not args.skip_clone or not Path(args.clone_dir).exists():
        RepoFetcher(args.clone_dir).fetch(args.repo_url)
    else:
        logger.info("Using existing repo at %s", args.clone_dir)

    # 2. Static analysis
    logger.info("Running static analysis...")
    code_data = CodeProcessor(args.clone_dir).process()
    logger.info(
        "Processed %d files, %d LOC, %d methods",
        code_data["stats"]["total_files"],
        code_data["stats"]["total_code_lines"],
        code_data["stats"]["total_methods"],
    )

    # 3. LLM analysis
    logger.info("Running LLM analysis...")
    analysis = CodeAnalyzer().analyze(code_data)

    # 4. Write output (strip raw file content before serializing)
    for f in code_data["files"]:
        f.pop("content", None)

    output = build_output(code_data, analysis)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    logger.info("Done. Output written to %s", args.output)


if __name__ == "__main__":
    main()
