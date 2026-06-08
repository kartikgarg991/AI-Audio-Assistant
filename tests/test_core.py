import unittest

from app.ai import split_text
from app.session_store import clear_session, get_json, set_json, touch_session


class CoreTests(unittest.TestCase):
    def test_split_text_preserves_content(self):
        source = " ".join(f"word-{index}" for index in range(200))
        chunks = split_text(source, size=120, overlap=0)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), source)

    def test_memory_session_round_trip(self):
        session_id = "test-session"
        touch_session(session_id)
        set_json("transcript", session_id, {"text": "hello"})

        self.assertEqual(get_json("transcript", session_id), {"text": "hello"})
        clear_session(session_id)
        self.assertIsNone(get_json("transcript", session_id))


if __name__ == "__main__":
    unittest.main()
