import sqlite3
import json
from datetime import datetime, timezone
from config.settings import DB_PATH

_conn = None

def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn

def _table_exists(conn, name):
    r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return r is not None

def _column_exists(conn, table, column):
    try:
        conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        return True
    except:
        return False

def init_db():
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT,
        role TEXT,
        goals TEXT,
        timezone TEXT DEFAULT 'UTC',
        peak_hours TEXT DEFAULT '9-17',
        briefing_time TEXT DEFAULT '08:00',
        voice TEXT DEFAULT 'female',
        calendar_token TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        priority TEXT DEFAULT 'medium',
        due_date TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        date TEXT,
        time TEXT,
        duration INTEGER DEFAULT 60,
        is_shared INTEGER DEFAULT 0,
        people TEXT,
        calendar_event_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        remind_at TEXT,
        sent INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        message TEXT,
        intent TEXT,
        entities TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        content TEXT,
        category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ideas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS pending_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action_type TEXT,
        action_data TEXT,
        question TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS briefing_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS eod_check_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS habit_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        habit_type TEXT,
        hour INTEGER,
        count INTEGER DEFAULT 1,
        date TEXT
    )""")
    if _table_exists(conn, "habit_tracking") and not _column_exists(conn, "habit_tracking", "habit_type"):
        cur.execute("DROP TABLE habit_tracking")
        cur.execute("""
        CREATE TABLE habit_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            habit_type TEXT,
            hour INTEGER,
            count INTEGER DEFAULT 1,
            date TEXT
        )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS shared_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id INTEGER,
        user_id INTEGER,
        status TEXT DEFAULT 'pending'
    )""")

    conn.commit()

# Users
def add_user(user_id, name, role="", goals="", timezone="UTC", peak_hours="9-17", briefing_time="08:00", voice="female"):
    conn = _get_conn()
    conn.execute("INSERT OR REPLACE INTO users (user_id, name, role, goals, timezone, peak_hours, briefing_time, voice) VALUES (?,?,?,?,?,?,?,?)",
                 (user_id, name, role, goals, timezone, peak_hours, briefing_time, voice))
    conn.commit()

def get_user(user_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None

def get_user_by_name(name):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
    return dict(row) if row else None

def get_all_users():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM users").fetchall()
    return [dict(r) for r in rows]

def update_user_voice(user_id, voice):
    conn = _get_conn()
    conn.execute("UPDATE users SET voice = ? WHERE user_id = ?", (voice, user_id))
    conn.commit()

def update_user_calendar_token(user_id, token_json):
    conn = _get_conn()
    conn.execute("UPDATE users SET calendar_token = ? WHERE user_id = ?", (token_json, user_id))
    conn.commit()

# Tasks
def add_task(user_id, title, priority="medium", due_date=None):
    conn = _get_conn()
    cur = conn.execute("INSERT INTO tasks (user_id, title, priority, due_date) VALUES (?,?,?,?)",
                       (user_id, title, priority, due_date))
    conn.commit()
    return cur.lastrowid

def get_tasks(user_id, status=None):
    conn = _get_conn()
    sql = "SELECT * FROM tasks WHERE user_id = ?"
    params = [user_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]

def get_pending_tasks(user_id):
    return get_tasks(user_id, status="pending")

def complete_task(task_id):
    conn = _get_conn()
    conn.execute("UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,))
    conn.commit()

def update_task_priority(task_id, priority):
    conn = _get_conn()
    conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (priority, task_id))
    conn.commit()

def update_task_due_date(task_id, due_date):
    conn = _get_conn()
    conn.execute("UPDATE tasks SET due_date = ? WHERE id = ?", (due_date, task_id))
    conn.commit()

def update_task_status(task_id, status):
    conn = _get_conn()
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()

def update_task_title(task_id, title):
    conn = _get_conn()
    conn.execute("UPDATE tasks SET title = ? WHERE id = ?", (title, task_id))
    conn.commit()

def delete_task(user_id, task_id=None, title=None):
    conn = _get_conn()
    if task_id:
        conn.execute("DELETE FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id))
    elif title:
        conn.execute("DELETE FROM tasks WHERE title LIKE ? AND user_id = ?", (f"%{title}%", user_id))
    conn.commit()

def delete_all_tasks(user_id):
    conn = _get_conn()
    conn.execute("DELETE FROM tasks WHERE user_id = ?", (user_id,))
    conn.commit()

# Events
def add_event(user_id, title, date, time, duration=60, is_shared=0, people="[]", calendar_event_id=None):
    conn = _get_conn()
    cur = conn.execute("INSERT INTO events (user_id, title, date, time, duration, is_shared, people, calendar_event_id) VALUES (?,?,?,?,?,?,?,?)",
                       (user_id, title, date, time, duration, is_shared, people, calendar_event_id))
    conn.commit()
    return cur.lastrowid

def get_events(user_id, date=None):
    conn = _get_conn()
    sql = """SELECT e.* FROM events e
             LEFT JOIN shared_events se ON e.id = se.event_id
             WHERE (e.user_id = ? OR (se.user_id = ? AND se.status = 'accepted'))"""
    params = [user_id, user_id]
    if date:
        sql += " AND e.date = ?"
        params.append(date)
    sql += " ORDER BY e.date, e.time"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]

def delete_event(user_id, event_id=None, title=None):
    conn = _get_conn()
    if event_id:
        conn.execute("DELETE FROM events WHERE id = ? AND user_id = ?", (event_id, user_id))
    elif title:
        conn.execute("DELETE FROM events WHERE title LIKE ? AND user_id = ?", (f"%{title}%", user_id))
    conn.commit()

# Shared events
def add_shared_event(event_id, user_id):
    conn = _get_conn()
    conn.execute("INSERT OR IGNORE INTO shared_events (event_id, user_id, status) VALUES (?,?,'pending')", (event_id, user_id))
    conn.commit()

def accept_shared_event(event_id, user_id):
    conn = _get_conn()
    conn.execute("UPDATE shared_events SET status = 'accepted' WHERE event_id = ? AND user_id = ?", (event_id, user_id))
    conn.commit()

# Reminders
def add_reminder(user_id, title, remind_at):
    conn = _get_conn()
    conn.execute("INSERT INTO reminders (user_id, title, remind_at) VALUES (?,?,?)", (user_id, title, remind_at))
    conn.commit()

def get_user_reminders(user_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM reminders WHERE user_id = ? AND sent = 0 ORDER BY remind_at", (user_id,)).fetchall()
    return [dict(r) for r in rows]

def get_due_reminders(now_iso):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM reminders WHERE sent = 0 AND remind_at <= ? ORDER BY remind_at", (now_iso,)).fetchall()
    return [dict(r) for r in rows]

def mark_reminder_sent(reminder_id):
    conn = _get_conn()
    conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))
    conn.commit()

def delete_reminder(user_id, reminder_id=None, title=None):
    conn = _get_conn()
    if reminder_id:
        conn.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
    elif title:
        conn.execute("DELETE FROM reminders WHERE title LIKE ? AND user_id = ?", (f"%{title}%", user_id))
    conn.commit()

# Chat history
def add_chat_message(user_id, role, message, intent="", entities=None):
    conn = _get_conn()
    conn.execute("INSERT INTO chat_history (user_id, role, message, intent, entities) VALUES (?,?,?,?,?)",
                 (user_id, role, message, intent, json.dumps(entities) if entities else None))
    conn.commit()

def get_chat_history(user_id, limit=20):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]

def get_last_assistant_message(user_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM chat_history WHERE user_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
    return dict(row) if row else None

# Memories
def add_memory(user_id, content, category="general"):
    conn = _get_conn()
    conn.execute("INSERT INTO memories (user_id, content, category) VALUES (?,?,?)", (user_id, content, category))
    conn.commit()

def get_memories(user_id, topic=""):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM memories WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 10",
                        (user_id, f"%{topic}%")).fetchall()
    return [dict(r) for r in rows]

# Ideas
def add_idea(user_id, title):
    conn = _get_conn()
    conn.execute("INSERT INTO ideas (user_id, title) VALUES (?,?)", (user_id, title))
    conn.commit()

def get_ideas(user_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM ideas WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]

def delete_idea(user_id, idea_id):
    conn = _get_conn()
    conn.execute("DELETE FROM ideas WHERE id = ? AND user_id = ?", (idea_id, user_id))
    conn.commit()

# Pending actions
def add_pending_action(user_id, action_type, action_data, question=""):
    conn = _get_conn()
    conn.execute("INSERT INTO pending_actions (user_id, action_type, action_data, question) VALUES (?,?,?,?)",
                 (user_id, action_type, json.dumps(action_data), question))
    conn.commit()

def get_pending_actions(user_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM pending_actions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user_id,)).fetchall()
    return [dict(r) for r in rows]

def clear_pending_actions(user_id):
    conn = _get_conn()
    conn.execute("DELETE FROM pending_actions WHERE user_id = ?", (user_id,))
    conn.commit()

def clear_pending_action(user_id, action_type):
    conn = _get_conn()
    conn.execute("DELETE FROM pending_actions WHERE user_id = ? AND action_type = ?", (user_id, action_type))
    conn.commit()

# Briefing log
def mark_briefing_sent(user_id, date):
    conn = _get_conn()
    conn.execute("INSERT INTO briefing_log (user_id, date) VALUES (?,?)", (user_id, date))
    conn.commit()

def was_briefing_sent_today(user_id, date):
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM briefing_log WHERE user_id = ? AND date = ? LIMIT 1", (user_id, date)).fetchone()
    return row is not None

# EOD check log
def mark_eod_sent(user_id, date):
    conn = _get_conn()
    conn.execute("INSERT INTO eod_check_log (user_id, date) VALUES (?,?)", (user_id, date))
    conn.commit()

def was_eod_sent_today(user_id, date):
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM eod_check_log WHERE user_id = ? AND date = ? LIMIT 1", (user_id, date)).fetchone()
    return row is not None

# Habits
def track_habit(user_id, habit_type, hour=None, date=None):
    conn = _get_conn()
    if hour is None:
        hour = datetime.now(timezone.utc).hour
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute("INSERT INTO habit_tracking (user_id, habit_type, hour, date) VALUES (?,?,?,?)",
                 (user_id, habit_type, hour, date))
    conn.commit()

def get_habits(user_id, habit_type):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM habit_tracking WHERE user_id = ? AND habit_type = ? ORDER BY date DESC", (user_id, habit_type)).fetchall()
    return [dict(r) for r in rows]
