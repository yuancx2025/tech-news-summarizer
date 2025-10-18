import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "langchain_google_genai" not in sys.modules:
    fake_module = types.ModuleType("langchain_google_genai")

    class _FakeEmbeddings:
        def __init__(self, *_, **__):
            pass

    class _FakeChat:
        def __init__(self, *_, **__):
            pass

        def invoke(self, *_args, **__):
            return types.SimpleNamespace(content="")

    fake_module.GoogleGenerativeAIEmbeddings = _FakeEmbeddings
    fake_module.ChatGoogleGenerativeAI = _FakeChat
    sys.modules["langchain_google_genai"] = fake_module

if "dateparser" not in sys.modules:
    fake_dateparser = types.ModuleType("dateparser")

    def _noop_parse(*_, **__):
        return None

    fake_dateparser.parse = _noop_parse
    sys.modules["dateparser"] = fake_dateparser

    fake_search = types.ModuleType("dateparser.search")

    def _noop_search(*_, **__):
        return []

    fake_search.search_dates = _noop_search
    sys.modules["dateparser.search"] = fake_search
