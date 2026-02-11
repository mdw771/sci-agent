from pathlib import Path

from sciagent.skill import extract_markdown_image_paths


def test_extract_markdown_image_paths_supports_common_markdown_forms(tmp_path: Path):
    markdown_path = tmp_path / "docs" / "intro.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("placeholder", encoding="utf-8")

    markdown = "\n".join(
        [
            "![absolute](/tmp/abs.png)",
            "![relative](images/rel.png)",
            "![angle](<figures/plot.png>)",
            "![titled](assets/final.png \"result figure\")",
            "![remote](https://example.com/x.png)",
        ]
    )

    actual = extract_markdown_image_paths(markdown, markdown_path=markdown_path)

    expected = [
        "/tmp/abs.png",
        str((markdown_path.parent / "images/rel.png").resolve()),
        str((markdown_path.parent / "figures/plot.png").resolve()),
        str((markdown_path.parent / "assets/final.png").resolve()),
    ]
    assert actual == expected


def test_extract_markdown_image_paths_without_markdown_path_keeps_relative_paths():
    markdown = "![a](img/a.png)\n![b](../img/b.png)"

    actual = extract_markdown_image_paths(markdown)

    assert actual == ["img/a.png", "../img/b.png"]
