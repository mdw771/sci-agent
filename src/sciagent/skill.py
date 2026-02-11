from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

from sciagent.tool.base import BaseTool, ToolReturnType, tool

logger = logging.getLogger(__name__)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    tool_name: str
    path: str


class SkillTool(BaseTool):
    def __init__(
        self,
        metadata: SkillMetadata,
        *,
        max_doc_bytes: int = 200_000,
        require_approval: bool = False,
        **kwargs: Any,
    ) -> None:
        self.metadata = metadata
        self.max_doc_bytes = max_doc_bytes
        super().__init__(require_approval=require_approval, **kwargs)

    def build(self, *args: Any, **kwargs: Any) -> None:
        self.tool_name_overrides = {"fetch_skill_docs": self.metadata.tool_name}

    @tool(name="fetch_skill_docs", return_type=ToolReturnType.DICT)
    def fetch_skill_docs(self) -> Dict[str, Any]:
        """Returns the documentation files for this skill."""
        files, skipped, images_by_file = collect_skill_docs(
            Path(self.metadata.path), max_doc_bytes=self.max_doc_bytes
        )
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "path": self.metadata.path,
            "files": files,
            "images_by_file": images_by_file,
            "skipped_files": skipped,
        }


def load_skills(skill_dirs: Sequence[str]) -> List[SkillMetadata]:
    skills: List[SkillMetadata] = []
    seen_tool_names: set[str] = set()

    for base_dir in skill_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning("Skill directory not found: %s", base_dir)
            continue

        if (base_path / "SKILL.md").exists():
            skill_paths = [base_path]
        else:
            skill_paths = sorted({path.parent for path in base_path.rglob("SKILL.md")})

        for skill_path in skill_paths:
            metadata = parse_skill_metadata(skill_path)
            if metadata is None:
                continue
            tool_name = ensure_unique_tool_name(metadata.tool_name, seen_tool_names)
            if tool_name != metadata.tool_name:
                metadata = SkillMetadata(
                    name=metadata.name,
                    description=metadata.description,
                    tool_name=tool_name,
                    path=metadata.path,
                )
            skills.append(metadata)

    return skills


def parse_skill_metadata(skill_dir: Path) -> SkillMetadata | None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    text = skill_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    frontmatter, body_lines = parse_frontmatter(lines)

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    name_index = 0
    if not name:
        name, name_index = extract_name(body_lines)
    if not name:
        name = skill_dir.name
        name_index = 0
    if not description:
        description = extract_description(body_lines, name_index)
    if not description:
        description = "No description provided."

    tool_name = make_tool_name(name)
    return SkillMetadata(
        name=name,
        description=description,
        tool_name=tool_name,
        path=str(skill_dir),
    )


def parse_frontmatter(lines: List[str]) -> Tuple[Dict[str, str], List[str]]:
    if not lines or lines[0].strip() != "---":
        return {}, lines

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter_lines = lines[1:index]
            body_lines = lines[index + 1 :]
            return read_frontmatter(frontmatter_lines), body_lines

    return {}, lines


def read_frontmatter(lines: List[str]) -> Dict[str, str]:
    frontmatter: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"").strip("'")
    return frontmatter


def extract_name(lines: List[str]) -> Tuple[str | None, int]:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("#").strip()
            if name:
                return name, index

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            name = stripped.split(":", 1)[1].strip()
            if name:
                return name, index

    return None, 0


def extract_description(lines: List[str], name_index: int) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("description:"):
            return stripped.split(":", 1)[1].strip()

    return extract_first_paragraph(lines, start_index=name_index + 1)


def extract_first_paragraph(lines: List[str], start_index: int) -> str:
    paragraphs: List[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if paragraphs:
                break
            continue
        paragraphs.append(stripped)
    return " ".join(paragraphs)


def make_tool_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = "skill"
    return f"skill-{slug}"


def ensure_unique_tool_name(tool_name: str, seen: set[str]) -> str:
    if tool_name not in seen:
        seen.add(tool_name)
        return tool_name

    index = 2
    while True:
        candidate = f"{tool_name}-{index}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        index += 1


def collect_skill_docs(
    skill_dir: Path,
    *,
    max_doc_bytes: int = 200_000,
) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]]]:
    files: Dict[str, str] = {}
    skipped: List[str] = []
    images_by_file: Dict[str, List[str]] = {}

    doc_paths: list[Path] = []
    for path in skill_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() != ".md":
            continue

        doc_paths.append(path)

    def path_sort_key(path: Path) -> tuple[int, str]:
        relative_path = str(path.relative_to(skill_dir))
        if relative_path.lower() == "skill.md":
            return (0, relative_path.lower())
        return (1, relative_path.lower())

    for path in sorted(doc_paths, key=path_sort_key):

        relative_path = str(path.relative_to(skill_dir))
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append(relative_path)
            continue

        if size > max_doc_bytes:
            skipped.append(relative_path)
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append(relative_path)
            continue

        files[relative_path] = content
        images_by_file[relative_path] = extract_markdown_image_paths(
            content,
            markdown_path=path,
        )

    return files, skipped, images_by_file


def extract_markdown_image_paths(
    markdown_content: str,
    *,
    markdown_path: Path | None = None,
) -> List[str]:
    """Extract image paths from markdown image tags.

    Paths in the form of `![alt](path)` are supported. When `markdown_path`
    is provided, relative paths are resolved against the markdown file parent.
    Remote URLs are ignored.
    """
    image_paths: List[str] = []
    for match in MARKDOWN_IMAGE_PATTERN.finditer(markdown_content):
        raw_target = match.group(1).strip()
        if not raw_target:
            continue

        # Allow markdown links that wrap the path in angle brackets.
        if raw_target.startswith("<") and raw_target.endswith(">"):
            raw_target = raw_target[1:-1].strip()

        # Handle optional markdown title: ![alt](path "title")
        if " " in raw_target:
            raw_target = raw_target.split(" ", 1)[0].strip()

        if "://" in raw_target:
            continue

        resolved_path = raw_target
        if markdown_path is not None and not Path(raw_target).is_absolute():
            resolved_path = str((markdown_path.parent / raw_target).resolve())

        image_paths.append(resolved_path)

    return image_paths
