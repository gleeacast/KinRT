#!/usr/bin/env python3
"""Validate the KinRT source package and maintain deterministic manifests."""

from __future__ import annotations

import argparse
import ast
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys
import tokenize
import tomllib
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("policy", "script")
CODE_SUFFIXES = {".py", ".sh", ".bash", ".yml", ".yaml", ".toml", ".json", ".html", ".css", ".js"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
FORBIDDEN_FILE = re.compile(r"(?:\.py[co]$|\.bak|\.orig$|\.rej$|\.pid$|\.log$)", re.IGNORECASE)
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".venv", "checkpoints", "wandb"}
SECRET_MARKERS = (
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    "-----BEGIN " + "RSA PRIVATE KEY-----",
    "Key" + "Pair-",
    "10.103." + "92.120",
)
EXPECTED_CONFIGS = {
    "kinrt_full",
    "kinrt_full_pi0",
    "kinrt_lora",
    "pi05_full_diyrobot",
    "pi05_lora_diyrobot",
    "pi0_full_diyrobot",
    "pi0_lora_diyrobot",
    "kinrt_full_diyrobot",
    "kinrt_lora_diyrobot",
    "kinrt_full_pi0_diyrobot",
    "kinrt_lora_pi0_diyrobot",
    "kinrt_adamoe_diyrobot",
}
EXPECTED_PAPER_MODELS = {
    "OpenVLA",
    "RDT-1B",
    "pi0-Full",
    "pi0-LoRA",
    "pi0.5-Full",
    "pi0.5-LoRA",
    "Hi-MoE",
    "AdaMoE",
    "KinRT-OpenVLA",
    "KinRT-Full (pi0)",
    "KinRT-LoRA (pi0)",
    "KinRT-AdaMoE",
    "KinRT-Full",
    "KinRT-LoRA",
}
EXPECTED_SITE_FILES = {
    "index.html",
    "getting-started.html",
    "method.html",
    "robotwin.html",
    "diyrobot.html",
    "real-robot.html",
    "reference.html",
    "404.html",
    "styles.css",
    "app.js",
    ".nojekyll",
}


class SiteHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for attribute in ("href", "src"):
            if values.get(attribute):
                self.references.append((attribute, values[attribute]))


def parse_site_html(path: Path) -> SiteHTMLParser:
    parser = SiteHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def validate_site() -> list[str]:
    errors: list[str] = []
    site_root = ROOT / "docs"
    missing = sorted(name for name in EXPECTED_SITE_FILES if not (site_root / name).exists())
    if missing:
        errors.append(f"missing website files: {missing}")

    html_paths = files_under(site_root)
    html_paths = [path for path in html_paths if path.suffix.lower() == ".html"]
    parsed: dict[Path, SiteHTMLParser] = {}
    for path in html_paths:
        try:
            parsed[path.resolve()] = parse_site_html(path)
        except (UnicodeDecodeError, Exception) as exc:
            errors.append(f"HTML parse failure: {path.relative_to(ROOT).as_posix()}: {exc}")

    for source, parser in parsed.items():
        for attribute, value in parser.references:
            parts = urlsplit(value)
            if parts.scheme or parts.netloc or value.startswith(("mailto:", "tel:", "data:", "javascript:")):
                continue

            raw_path = unquote(parts.path)
            if raw_path:
                target = (source.parent / raw_path).resolve()
            else:
                target = source

            try:
                target.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(
                    f"website {attribute} escapes release root: {source.relative_to(ROOT).as_posix()} -> {value}"
                )
                continue

            if not target.exists():
                errors.append(f"broken website {attribute}: {source.relative_to(ROOT).as_posix()} -> {value}")
                continue

            if parts.fragment and target.suffix.lower() == ".html":
                target_parser = parsed.get(target)
                if target_parser is not None and parts.fragment not in target_parser.ids:
                    errors.append(
                        f"broken website fragment: {source.relative_to(ROOT).as_posix()} -> {value}"
                    )

    return errors


def files_under(path: Path):
    return sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    return [path for source_root in SOURCE_ROOTS for path in files_under(ROOT / source_root)]


def manifest_text() -> str:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in source_files()]
    return "\n".join(lines) + "\n"


def train_config_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CONFIGS" for target in node.targets)
    )
    if not isinstance(assignment.value, ast.List):
        raise ValueError("_CONFIGS must be a literal list")

    names: set[str] = set()
    for element in assignment.value.elts:
        if not isinstance(element, ast.Call):
            raise ValueError("_CONFIGS contains a non-call entry")
        name = next(
            (
                keyword.value.value
                for keyword in element.keywords
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
        if not isinstance(name, str):
            raise ValueError("TrainConfig entry has no literal name")
        if name in names:
            raise ValueError(f"duplicate TrainConfig name: {name}")
        names.add(name)
    return names


def python_comment_count(text: str) -> int:
    return sum(
        token.type == tokenize.COMMENT
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    )


def python_docstring_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            count += 1
    return count


def validate() -> list[str]:
    errors: list[str] = []
    for path in files_under(ROOT):
        rel = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in rel.parts) or FORBIDDEN_FILE.search(path.name):
            errors.append(f"forbidden artifact: {rel.as_posix()}")
            continue

        if path.suffix == ".py":
            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text, filename=str(rel))
                python_comment_count(text)
                python_docstring_count(tree)
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"python parse failure: {rel.as_posix()}: {exc}")

        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"JSON parse failure: {rel.as_posix()}: {exc}")

        if path.suffix == ".toml":
            try:
                tomllib.loads(path.read_text(encoding="utf-8"))
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"TOML parse failure: {rel.as_posix()}: {exc}")

        if path.suffix.lower() in CODE_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"UTF-8 decode failure: {rel.as_posix()}: {exc}")
                continue
            if CJK.search(text):
                errors.append(f"CJK text in active code/config: {rel.as_posix()}")

        if path.suffix.lower() in CODE_SUFFIXES | {".md", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in SECRET_MARKERS:
                if marker in text:
                    errors.append(f"private credential/host marker in {rel.as_posix()}: {marker}")

    config_path = ROOT / "policy" / "pi05" / "src" / "openpi" / "training" / "config.py"
    try:
        all_configs = train_config_names(config_path)
    except (SyntaxError, UnicodeDecodeError, ValueError, StopIteration) as exc:
        errors.append(f"config registry parse failure: {exc}")
        all_configs = set()
    if all_configs != EXPECTED_CONFIGS:
        errors.append(
            f"unexpected config registry: missing={sorted(EXPECTED_CONFIGS - all_configs)}, "
            f"extra={sorted(all_configs - EXPECTED_CONFIGS)}"
        )

    mapping_path = ROOT / "configs" / "PAPER_MODELS.json"
    try:
        models = json.loads(mapping_path.read_text(encoding="utf-8"))["models"]
        paper_names = [item["paper_model"] for item in models]
        if len(paper_names) != len(set(paper_names)):
            errors.append("duplicate paper model in configs/PAPER_MODELS.json")
        if set(paper_names) != EXPECTED_PAPER_MODELS:
            errors.append(
                "unexpected paper model set: "
                f"missing={sorted(EXPECTED_PAPER_MODELS - set(paper_names))}, "
                f"extra={sorted(set(paper_names) - EXPECTED_PAPER_MODELS)}"
            )
        mapped_configs = {item["release_config"] for item in models if item["release_config"] is not None}
        if len(mapped_configs) != 9:
            errors.append(f"expected 9 OpenPI paper configs, found {len(mapped_configs)}")
        if not mapped_configs <= all_configs:
            errors.append(f"paper mapping references missing configs: {sorted(mapped_configs - all_configs)}")
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"paper mapping parse failure: {exc}")

    errors.extend(validate_site())

    return errors


def write_manifests() -> None:
    manifest_dir = ROOT / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "KINRT_SOURCE_SHA256.txt").write_text(manifest_text(), encoding="utf-8", newline="\n")


def check_manifests() -> list[str]:
    errors: list[str] = []
    path = ROOT / "manifests" / "KINRT_SOURCE_SHA256.txt"
    expected = manifest_text()
    if not path.exists():
        errors.append(f"missing manifest: {path.relative_to(ROOT).as_posix()}")
    elif path.read_text(encoding="utf-8") != expected:
        errors.append(f"stale manifest: {path.relative_to(ROOT).as_posix()}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--check-manifests", action="store_true")
    args = parser.parse_args()

    if args.write_manifests:
        write_manifests()

    errors = validate()
    if args.check_manifests:
        errors.extend(check_manifests())

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    python_count = sum(1 for path in files_under(ROOT) if path.suffix == ".py")
    python_comments = 0
    python_docstrings = 0
    for path in files_under(ROOT):
        if path.suffix == ".py":
            text = path.read_text(encoding="utf-8")
            python_comments += python_comment_count(text)
            python_docstrings += python_docstring_count(ast.parse(text))
    source_count = len(source_files())
    print(
        f"Validation passed: {python_count} Python files, {python_comments} Python comments, "
        f"{python_docstrings} docstrings, {source_count} source files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
