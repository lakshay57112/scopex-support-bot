import json
from dataclasses import dataclass
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    url: str
    content: str
    questions: tuple[str, ...] = ()  # common ways customers ask about this

    @property
    def text(self) -> str:
        # Title and example questions go in front so short or casual
        # questions still match the right article
        asked = " ".join(self.questions)
        return f"{self.title}. {asked} {self.content}"


def load_articles(path: Path = KB_PATH) -> list[Article]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Article(**{**item, "questions": tuple(item.get("questions", []))}) for item in raw]
