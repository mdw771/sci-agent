import sqlite3

from sciagent.task_manager.base import BaseTaskManager


def _create_message_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (timestamp TEXT, role TEXT, content TEXT, tool_calls TEXT, image TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS status (id INTEGER PRIMARY KEY, user_input_requested INTEGER)"
        )
        conn.execute(
            "INSERT INTO status (id, user_input_requested) VALUES (1, 0)"
        )
        rows = [
            ("1", "system", "DB system message", None, None),
            ("2", "user_webui", "Hello from DB", None, None),
            (
                "3",
                "assistant",
                "Calling tool",
                'call_1: demo_tool\nArguments: {"x": 1}\n',
                None,
            ),
            ("4", "tool", "Tool response", None, None),
            ("5", "user_webui", "Here is an image\n<image>\n", None, "data:image/png;base64,FAKE"),
        ]
        conn.executemany(
            "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_fill_context_with_message_db(tmp_path):
    db_path = tmp_path / "messages.db"
    _create_message_db(str(db_path))

    try:
        manager = BaseTaskManager(
            message_db_path=str(db_path),
            fill_context_with_message_db=True,
            build=False,
        )
        manager.context = [{"role": "system", "content": "Initial system"}]
        manager.full_history = [{"role": "system", "content": "Initial system"}]

        manager.build_db()

        assert manager.context[0]["role"] == "system"
        assert manager.context[0]["content"] == "Initial system"

        inserted = manager.context[1:]
        assert [msg["role"] for msg in inserted] == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert inserted[2]["tool_calls"][0]["id"] == "call_1"
        assert inserted[2]["tool_calls"][0]["function"]["name"] == "demo_tool"
        assert inserted[3]["tool_call_id"] == "call_1"

        image_message = inserted[4]
        assert isinstance(image_message["content"], list)
        assert image_message["content"][0]["type"] == "text"
        assert image_message["content"][0]["text"] == "Here is an image"
        assert image_message["content"][1]["type"] == "image_url"
    finally:
        if db_path.exists():
            db_path.unlink()
