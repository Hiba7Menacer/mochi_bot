"""Mochi Telegram Bot."""
import os
import re
import json
import asyncio
import tempfile
import traceback
from datetime import datetime, timedelta, timezone

import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config.settings import TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODEL, GOOGLE_CLIENT_ID
from brain.brain import Brain
from memory.db import (
    init_db, add_user, get_user, get_user_by_name, get_all_users,
    update_user_voice, update_user_calendar_token,
    add_task, get_tasks, get_pending_tasks, complete_task, update_task_priority, update_task_due_date, update_task_status, update_task_title, delete_task, delete_all_tasks,
    add_event, get_events, delete_event,
    add_reminder, get_user_reminders, get_due_reminders, mark_reminder_sent, delete_reminder,
    add_memory, get_memories,
    add_idea, get_ideas, delete_idea,
    add_chat_message, get_chat_history, get_last_assistant_message,
    mark_briefing_sent, was_briefing_sent_today,
    mark_eod_sent, was_eod_sent_today,
    track_habit, get_habits,
    add_pending_action, get_pending_actions, clear_pending_actions, clear_pending_action,
    add_shared_event, accept_shared_event,
)
from tools.calendar import CalendarTool

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

brain = Brain()

user_setup = {}

SETUP_QUESTIONS = [
    ("name", "What is your name?"),
    ("role", "What is your role? (e.g., Founder, Manager, Student)"),
    ("goals", "What are your top 3 goals? (comma separated)"),
    ("timezone", "What is your timezone? (e.g., Africa/Algiers, Europe/Paris)"),
    ("peak_hours", "What are your peak productivity hours? (e.g., 9-12, 14-17)"),
    ("briefing_time", "What time should I send your morning briefing? (e.g., 08:00)"),
    ("voice", "What voice should I use? (male or female)")
]

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def _user_ctx(user):
    if not user:
        return None
    return {
        "name": user.get("name", ""),
        "role": user.get("role", ""),
        "timezone": user.get("timezone", "UTC"),
        "peak_hours": user.get("peak_hours", "9-17"),
        "briefing_time": user.get("briefing_time", "08:00"),
        "voice": user.get("voice", "female")
    }

async def _reply(update, text, user_context=None):
    await update.message.reply_text(text)

# ------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user = get_user(uid)
    if user:
        name = user.get("name", "there")
        await update.message.reply_text(
            f"Welcome back {name}! I'm Mochi, your executive assistant.\n\n"
            "Try: 'Meeting tomorrow at 2pm' or 'Remind me in 5 minutes to prepare lunch'.\n\n"
            "Commands: /tasks /events /reminders /brief /voice /help"
        )
    else:
        user_setup[uid] = {"step": 0, "data": {}}
        await update.message.reply_text(
            "Hi! I'm Mochi, your AI executive assistant. Let's set up your profile.\n\n" + SETUP_QUESTIONS[0][1]
        )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Mochi Commands:\n\n"
        "/start — Start or restart setup\n"
        "/tasks — List your tasks\n"
        "/events — Today's events\n"
        "/reminders — Upcoming reminders\n"
        "/brief — Morning briefing now\n"
        "/auth_calendar — Connect Google Calendar\n"
        "/calendar_code <code> — Complete Google auth\n"
        "/voice male|female — Change my voice\n"
        "/help — Show this help\n\n"
        "What I can do:\n"
        "• Schedule events: 'Meeting with Mouadh tomorrow at 7pm'\n"
        "• Set reminders: 'Remind me in 5 minutes to prepare lunch'\n"
        "• Add tasks: 'I need to finish the report by Friday'\n"
        "• Complete tasks: 'Mark workout as done' or 'Complete task 1'\n"
        "• Complete all: 'Mark all as done'\n"
        "• Change priority: 'Make report urgent' or 'Make task 1 high'\n"
        "• Move tasks: 'Move workout to tomorrow' or 'Move task 1 to Friday'\n"
        "• Rename: 'Rename task 2 to Submit report'\n"
        "• Search memory: 'What did I say about marketing?'\n"
        "• Save ideas: 'Idea: video testimonials'\n"
        "• Morning brief: 'Good morning'\n"
        "• Cancel: 'Cancel my meeting tomorrow' or 'Cancel task 1'\n"
        "• Clear all: 'Clear all tasks'\n"
        "• Focus time: 'Protect my focus time'\n\n"
        "I'm Mochi. I remember everything and learn your habits."
    )
    await update.message.reply_text(text)

async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    tasks = get_tasks(uid)
    if not tasks:
        await update.message.reply_text("You have no tasks. Say: 'I need to...' to add one.")
        return
    text = "Your Tasks:\n\n"
    for i, t in enumerate(tasks, 1):
        status = "[x]" if t.get("status") == "completed" else "[ ]"
        due = t.get("due_date", "No date")
        pri = t.get("priority", "medium").upper()
        text += f"{status} {i}. {t['title']} (Due: {due}, {pri})\n"
    await update.message.reply_text(text)

async def cmd_events(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    today = brain.today().strftime("%Y-%m-%d")
    now_time = brain.now().strftime("%H:%M")
    events = get_events(uid, date=today)
    if not events:
        await update.message.reply_text("No events for today.\nSay: 'Meeting tomorrow at 2pm' to add one.")
        return
    past, future = [], []
    for e in events:
        et = e.get("time", "00:00")
        (past if et < now_time else future).append(e)
    text = "Today's Events:\n\n"
    if future:
        text += "Upcoming:\n"
        for e in future:
            text += f"• {e.get('time','??:??')} — {e['title']}\n"
        text += "\n"
    if past:
        text += "Past:\n"
        for e in past:
            text += f"• ~~{e.get('time','??:??')} — {e['title']}~~\n"
    await update.message.reply_text(text)

async def cmd_remaining(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    today = brain.today().strftime("%Y-%m-%d")
    now_time = brain.now().strftime("%H:%M")
    events = get_events(uid, date=today)
    future = [e for e in events if e.get("time","00:00") >= now_time]
    text = "What you still have today:\n\n"
    if future:
        for e in future:
            text += f"• {e.get('time','??:??')} — {e['title']}\n"
    else:
        text += "No more events scheduled today!\n"
    pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
    if pending:
        text += f"\nPending Tasks ({len(pending)}):\n"
        for t in pending[:5]:
            text += f"• {t['title']}\n"
    await update.message.reply_text(text)

async def cmd_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    rems = get_user_reminders(uid)
    if not rems:
        await update.message.reply_text("No upcoming reminders.\nSay: 'Remind me in 5 minutes to...'")
        return
    text = "Your Reminders:\n\n"
    for r in rems:
        at = r['remind_at']
        try:
            dt = datetime.fromisoformat(at)
            at_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except:
            at_str = at
        text += f"• {r['title']} at {at_str}\n"
    await update.message.reply_text(text)

async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE, missed=False):
    uid = update.effective_user.id
    user = get_user(uid)
    name = user["name"] if user else "there"
    today = brain.today().strftime("%Y-%m-%d")
    now_time = brain.now().strftime("%H:%M")
    events = get_events(uid, date=today)
    past = [e for e in events if e.get("time","00:00") < now_time]
    future = [e for e in events if e.get("time","00:00") >= now_time]
    pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
    text = f"Sorry {name}, I missed your scheduled briefing!\n\n" if missed else f"Good morning {name}! Here's your briefing:\n\n"
    if future:
        text += f"Upcoming Events ({len(future)}):\n"
        for e in future:
            text += f"• {e.get('time','??:??')} — {e['title']}\n"
        text += "\n"
    else:
        text += "Upcoming Events: Nothing ahead!\n\n"
    if past:
        text += f"Completed Today ({len(past)}):\n"
        for e in past:
            text += f"• ~~{e.get('time','??:??')} — {e['title']}~~\n"
        text += "\n"
    if pending:
        text += f"Pending Tasks ({len(pending)}):\n"
        for t in pending[:5]:
            text += f"• {t['title']}\n"
    else:
        text += "Pending Tasks: All clear!\n"
    await update.message.reply_text(text)
    mark_briefing_sent(uid, today)

async def cmd_sync_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    cal = CalendarTool(user_id=uid)
    if not cal.is_authenticated():
        await update.message.reply_text("❌ Google Calendar not connected. Use /auth_calendar first.")
        return
    tasks = get_tasks(uid, status="pending")
    if not tasks:
        await update.message.reply_text("No pending tasks to sync.")
        return
    synced = 0
    failed = 0
    for t in tasks:
        try:
            cal.create_task(t["title"], due_date=t.get("due_date"))
            synced += 1
        except Exception as e:
            failed += 1
            print(f"[Sync Tasks] Failed for {t['title']}: {e}")
    await update.message.reply_text(f"✅ Synced {synced} tasks to Google Tasks." + (f"\n⚠️ {failed} failed." if failed else ""))

async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /voice male or /voice female")
        return
    choice = context.args[0].lower()
    if choice not in ["male", "female"]:
        await update.message.reply_text("Choose male or female.")
        return
    update_user_voice(uid, choice)
    await update.message.reply_text(f"Voice set to {choice}.")

async def cmd_auth_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    cal = CalendarTool(user_id=uid)
    try:
        url = cal.get_auth_url()
        if not url:
            await update.message.reply_text("Google Calendar is not configured in .env")
            return
        await update.message.reply_text(
            f"Authorize Google Calendar:\n{url}\n\n"
            "After authorizing, copy the code from the URL and send it here:\n"
            "/calendar_code YOUR_CODE\n\n"
            "If localhost doesn't work, set GOOGLE_REDIRECT_URI in .env to a real URL."
        )
    except Exception as e:
        await update.message.reply_text(f"Calendar auth error: {e}")

async def cmd_calendar_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    code = " ".join(context.args)
    if not code:
        await update.message.reply_text("Usage: /calendar_code YOUR_CODE")
        return
    # FIX: Extract code from full URL if pasted
    url_match = re.search(r"code=([^&\s]+)", code)
    if url_match:
        code = url_match.group(1)
        # URL-decode if needed
        from urllib.parse import unquote
        code = unquote(code)
    cal = CalendarTool(user_id=uid)
    try:
        cal.exchange_code(code)
        await update.message.reply_text("✅ Google Calendar & Tasks connected successfully!")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to connect: {e} Make sure you copied the full URL or just the code parameter.")

# ------------------------------------------------------------------
# MESSAGE HANDLERS
# ------------------------------------------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    msg = update.message.text

    if uid in user_setup:
        await _handle_setup(update, context)
        return

    user = get_user(uid)
    uctx = _user_ctx(user)
    chat_hist = get_chat_history(uid, limit=10)
    pending = get_pending_actions(uid)

    track_habit(uid, "message_received", hour=brain.now().hour)

    result = brain.process(msg, uctx, chat_hist, pending)
    intent = result.get("intent", "general_chat")
    entities = result.get("entities", {})
    resp_text = result.get("response_text", "I'm Mochi. How can I help?")

    add_chat_message(uid, "user", msg)
    action_resp = await _execute_intent(update, context, uid, intent, entities, resp_text, uctx)

    if action_resp:
        add_chat_message(uid, "assistant", action_resp, intent=intent, entities=entities)
        await _reply(update, action_resp, uctx)

async def _execute_intent(update, context, uid, intent, entities, resp_text, uctx):
    if intent == "multi_action":
        actions = entities.get("actions", [])
        out = []
        for a in actions:
            sub = await _execute_single_intent(update, context, uid, a.get("intent"), a.get("entities", {}), "", uctx)
            if sub:
                out.append(sub)
        clear_pending_actions(uid)
        return "\n\n".join(out) if out else resp_text

    r = await _execute_single_intent(update, context, uid, intent, entities, resp_text, uctx)
    if intent not in ("needs_followup", "ask_followup"):
        clear_pending_actions(uid)
    return r

async def _execute_single_intent(update, context, uid, intent, entities, resp_text, uctx):
    # --- CANCELLATIONS ---
    if intent in ("cancel_event", "cancel_task", "cancel_reminder", "cancel_search"):
        return await _handle_cancel(update, context, uid, intent, entities, uctx)

    # --- FOLLOW-UP / PENDING ---
    if intent in ("needs_followup", "ask_followup"):
        pending = entities.get("pending_action")
        if pending:
            add_pending_action(uid, pending, entities, question=resp_text)
        return resp_text or "Could you tell me more?"

    # --- MARK ALL DONE ---
    if intent == "mark_all_done":
        tasks = get_tasks(uid)
        pending = [t for t in tasks if t.get("status") != "completed"]
        if not pending:
            return "You have no pending tasks! Everything is already done. ✅"
        for t in pending:
            complete_task(t["id"])
            track_habit(uid, "task_completed", hour=brain.now().hour)
        # Show updated list
        remaining = [t for t in get_tasks(uid) if t.get("status") != "completed"]
        text = f"✅ Marked all {len(pending)} tasks as completed. Great work!\n\n"
        if remaining:
            text += "Remaining tasks:\n"
            for i, t in enumerate(remaining, 1):
                text += f"[ ] {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})\n"
        else:
            text += "🎉 All tasks are now complete!"
        return text

    # --- CLEAR ALL TASKS ---
    if intent == "clear_all_tasks":
        delete_all_tasks(uid)
        return "✅ All tasks cleared."

    # --- MARK TASK BY NUMBER ---
    if intent == "mark_task_by_number":
        nums = entities.get("numbers", [])
        all_tasks = get_tasks(uid)
        marked = []
        for n in nums:
            idx = n - 1
            if 0 <= idx < len(all_tasks):
                complete_task(all_tasks[idx]["id"])
                track_habit(uid, "task_completed", hour=brain.now().hour)
                marked.append(all_tasks[idx]["title"])
        if marked:
            # Show updated list
            remaining = [t for t in get_tasks(uid) if t.get("status") != "completed"]
            text = f"✅ Marked as completed: {', '.join(marked)}. Great work!\n\n"
            if remaining:
                text += "Updated task list:\n"
                for i, t in enumerate(remaining, 1):
                    text += f"[ ] {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})\n"
            else:
                text += "🎉 All tasks are now complete!"
            return text
        return "I couldn't find tasks with those numbers."

    # --- DELETE TASK BY NUMBER ---
    if intent == "delete_task_by_number":
        n = entities.get("number", 0)
        all_tasks = get_tasks(uid)
        idx = n - 1
        if 0 <= idx < len(all_tasks):
            title = all_tasks[idx]["title"]
            delete_task(uid, task_id=all_tasks[idx]["id"])
            # Show updated list
            remaining = get_tasks(uid)
            text = f"✅ Removed task {n}: {title}\n\n"
            if remaining:
                text += "Updated task list:\n"
                for i, t in enumerate(remaining, 1):
                    status = "[x]" if t.get("status") == "completed" else "[ ]"
                    text += f"{status} {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})\n"
            else:
                text += "Your task list is now empty."
            return text
        return f"Task {n} doesn't exist."

    # --- RENAME TASK ---
    if intent == "rename_task":
        n = entities.get("number", 0)
        new_title = entities.get("new_title", "")
        all_tasks = get_tasks(uid)
        idx = n - 1
        if 0 <= idx < len(all_tasks):
            old = all_tasks[idx]["title"]
            update_task_title(all_tasks[idx]["id"], new_title)
            return f"✅ Renamed task {n} from '{old}' to '{new_title}'."
        return f"Task {n} doesn't exist."

    # --- MOVE TASK BY NUMBER ---
    if intent == "move_task_by_number":
        n = entities.get("number", 0)
        date = entities.get("date")
        when = entities.get("when", "")
        all_tasks = get_tasks(uid)
        idx = n - 1
        if 0 <= idx < len(all_tasks):
            if date:
                update_task_due_date(all_tasks[idx]["id"], date)
                return f"📅 Moved task {n} '{all_tasks[idx]['title']}' to {date}."
            elif when:
                dt_info = brain._extract_time(when.lower(), uctx)
                if dt_info[3]:
                    update_task_due_date(all_tasks[idx]["id"], dt_info[0])
                    return f"📅 Moved task {n} '{all_tasks[idx]['title']}' to {dt_info[0]}."
            return f"📅 Moved task {n} '{all_tasks[idx]['title']}'."
        return f"Task {n} doesn't exist."

    # --- EDIT TASK TITLE (by title match) ---
    if intent == "edit_task_title":
        old_title = entities.get("old_title", "").lower().strip()
        new_title = entities.get("new_title", "").strip()
        tasks = get_tasks(uid)
        matching = [t for t in tasks if old_title in t["title"].lower() or t["title"].lower() in old_title]
        if matching:
            update_task_title(matching[0]["id"], new_title)
            # Show updated list
            remaining = get_tasks(uid)
            text = f"✅ Renamed task from '{matching[0]['title']}' to '{new_title}'"
            if remaining:
                text += "Updated task list:"
                for i, t in enumerate(remaining, 1):
                    status = "[x]" if t.get("status") == "completed" else "[ ]"
                    text += f"{status} {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})"
            return text
        return f"I couldn't find a task matching '{old_title}'."

    # --- EDIT TASK DATE (by title match) ---
    if intent == "edit_task_date":
        title = entities.get("title", "").lower().strip()
        new_date_str = entities.get("new_date", "")
        tasks = get_tasks(uid)
        matching = [t for t in tasks if title in t["title"].lower() or t["title"].lower() in title]
        if matching:
            dt_info = brain._extract_time(new_date_str.lower(), uctx)
            new_date = dt_info[0] if dt_info else brain.today().strftime("%Y-%m-%d")
            update_task_due_date(matching[0]["id"], new_date)
            return f"📅 Updated '{matching[0]['title']}' due date to {new_date}."
        return f"I couldn't find a task matching '{title}'."

    # --- EDIT EVENT (by title match) ---
    if intent == "edit_event":
        old_title = entities.get("old_title", "").lower().strip()
        new_title = entities.get("new_title", "").strip()
        events = get_events(uid)
        matching = [e for e in events if old_title in e["title"].lower() or e["title"].lower() in old_title]
        if matching:
            # We need to update the event title in DB
            from memory.db import _get_conn
            conn = _get_conn()
            conn.execute("UPDATE events SET title = ? WHERE id = ? AND user_id = ?", (new_title, matching[0]["id"], uid))
            conn.commit()
            return f"📅 Updated event from '{matching[0]['title']}' to '{new_title}'."
        return f"I couldn't find an event matching '{old_title}'."

    # --- SET WORKING HOURS FOR TODAY/TOMORROW ---
    if intent == "set_working_hours":
        day = entities.get("day", "today")
        start = entities.get("start", "9")
        end = entities.get("end", "17")
        # Store temporary working hours in memory
        add_memory(uid, f"working_hours_{day}:{start}-{end}", category="working_hours")
        return f"✅ Updated your working hours for {day} to {start}:00-{end}:00. I'll adjust your focus time and EOD reminders accordingly."

    # --- CHANGE PRIORITY BY NUMBER ---
    if intent == "change_priority_by_number":
        n = entities.get("number", 0)
        priority = entities.get("priority", "high")
        all_tasks = get_tasks(uid)
        idx = n - 1
        if 0 <= idx < len(all_tasks):
            update_task_priority(all_tasks[idx]["id"], priority)
            return f"🔥 Task {n} '{all_tasks[idx]['title']}' is now {priority.upper()} priority."
        return f"Task {n} doesn't exist."

    # --- SCHEDULE BRIEFING ---
    if intent == "schedule_briefing":
        bt = entities.get("briefing_time", "08:00")
        return f"✅ Got it! I'll send your morning briefing at {bt} every day. Sleep well!"

    # --- CHECK CALENDAR DATE ---
    if intent == "check_calendar_date":
        date = entities.get("date", brain.today().strftime("%Y-%m-%d"))
        events = get_events(uid, date=date)
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        if not events:
            return f"No events scheduled for {day_name}, {date}."
        text = f"Events for {day_name}, {date}:\n\n"
        for e in events:
            text += f"• {e.get('time','??:??')} — {e['title']}\n"
        return text

    # --- CHECK REMAINING TODAY ---
    if intent == "check_remaining_today":
        await cmd_remaining(update, context)
        return None

    # --- SET REMINDER ---
    if intent == "set_reminder":
        title = entities.get("title", "Reminder")
        remind_at = entities.get("remind_at")
        if remind_at:
            add_reminder(uid, title, remind_at)
            try:
                dt = datetime.fromisoformat(remind_at)
                disp = dt.strftime("%Y-%m-%d %H:%M UTC")
            except:
                disp = "soon"
            return f"⏰ Reminder set: I'll message you at {disp} about '{title}'."
        return "I couldn't set the reminder. When should I remind you?"

    # --- CREATE EVENT ---
    if intent == "create_event":
        title = entities.get("title", "Event")
        date = entities.get("date", brain.today().strftime("%Y-%m-%d"))
        time = entities.get("time", "09:00")
        duration = entities.get("duration", 60)
        people = entities.get("people", [])
        is_shared = 1 if people else 0
        people_json = json.dumps(people)
        eid = add_event(uid, title, date, time, duration, is_shared, people_json)

        # Auto-reminder for events: 1 day before, 1 hour before, 30 min before
        try:
            event_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            event_dt = event_dt.replace(tzinfo=timezone.utc)
            # Day before reminder
            day_before = event_dt - timedelta(days=1)
            if day_before > brain.now():
                add_reminder(uid, f"Tomorrow: {title}", day_before.strftime("%Y-%m-%dT%H:%M:%S"))
            # 1 hour before
            hour_before = event_dt - timedelta(hours=1)
            if hour_before > brain.now():
                add_reminder(uid, f"In 1 hour: {title}", hour_before.strftime("%Y-%m-%dT%H:%M:%S"))
            # 30 min before
            half_before = event_dt - timedelta(minutes=30)
            if half_before > brain.now():
                add_reminder(uid, f"In 30 min: {title}", half_before.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception as e:
            print(f"[Auto-reminder] Error: {e}")

        for person in people:
            other = get_user_by_name(person)
            if other and other["user_id"] != uid:
                add_shared_event(eid, other["user_id"])
                try:
                    await context.application.bot.send_message(
                        chat_id=other["user_id"],
                        text=f"📅 {uctx['name'] if uctx else 'Someone'} shared an event with you: {title} on {date} at {time}"
                    )
                except Exception as e:
                    print(f"[Shared] Notify error: {e}")

        cal = CalendarTool(user_id=uid)
        cal_msg = ""
        if cal.is_authenticated():
            try:
                tz = uctx.get("timezone", "UTC") if uctx else "UTC"
                cal.create_event(title, date, time, duration, timezone_str=tz)
                cal_msg = " (synced to Google Calendar)"
            except Exception as e:
                cal_msg = f"\n(Google Calendar sync failed: {e})"
        elif GOOGLE_CLIENT_ID and not cal.is_authenticated():
            cal_msg = "\n\nTip: Connect Google Calendar with /auth_calendar to sync automatically."

        return f"📅 Event scheduled:\n{title}\n{date} at {time}{cal_msg}"

    # --- CREATE TASK ---
        # --- CREATE TASK ---
    if intent == "create_task":
        title = entities.get("title", "Task")
        priority = entities.get("priority", "medium")
        due = entities.get("due_date", brain.today().strftime("%Y-%m-%d"))
        add_task(uid, title, priority=priority, due_date=due)
        
        # AUTO-SYNC to Google Tasks
        cal = CalendarTool(user_id=uid)
        sync_msg = ""
        if cal.is_authenticated():
            try:
                cal.create_task(title, due_date=due)
                sync_msg = " (synced to Google Tasks)"
            except Exception as e:
                print(f"[Auto-sync Tasks] Failed: {e}")
        
        habits = get_habits(uid, "task_completed")
        suggestion = ""
        if habits and len(habits) > 3:
            last_hour = habits[0]['hour']
            suggestion = f"\n💡 You usually complete tasks around {last_hour}:00. Want me to block focus time then?"
        return f"✅ Task added:\n{title}\nPriority: {priority.upper()} | Due: {due}{sync_msg}{suggestion}"
    # --- MARK TASK DONE (by title) ---
    if intent == "mark_task_done":
        title = entities.get("title", "").lower().strip()
        tasks = get_tasks(uid)
        matching = [t for t in tasks if title in t["title"].lower() or t["title"].lower() in title]
        if matching:
            complete_task(matching[0]["id"])
            track_habit(uid, "task_completed", hour=brain.now().hour)
            # Show updated list
            remaining = [t for t in get_tasks(uid) if t.get("status") != "completed"]
            text = f"✅ Marked as completed: {matching[0]['title']}. Great work!"
            if remaining:
                text += "Updated task list:"
                for i, t in enumerate(remaining, 1):
                    text += f"[ ] {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})"
            else:
                text += "🎉 All tasks are now complete!"
            return text
        return f"I couldn't find a task matching '{title}'."

    # --- CHANGE TASK PRIORITY (by title) ---
    if intent == "change_task_priority":
        title = entities.get("title", "").lower().strip()
        priority = entities.get("priority", "high")
        tasks = get_tasks(uid)
        matching = [t for t in tasks if title in t["title"].lower() or t["title"].lower() in title]
        if matching:
            update_task_priority(matching[0]["id"], priority)
            return f"🔥 Priority updated: '{matching[0]['title']}' is now {priority.upper()}."
        return f"I couldn't find a task matching '{title}'."

    # --- MOVE TASK (by title) ---
    if intent == "move_task":
        title = entities.get("title", "").lower().strip()
        when = entities.get("when", "tomorrow")
        tasks = get_tasks(uid)
        matching = [t for t in tasks if title in t["title"].lower() or t["title"].lower() in title]
        if matching:
            new_date = brain.today().strftime("%Y-%m-%d")
            if when == "tomorrow":
                new_date = (brain.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif when == "next week":
                new_date = (brain.today() + timedelta(weeks=1)).strftime("%Y-%m-%d")
            update_task_due_date(matching[0]["id"], new_date)
            return f"📅 Moved '{matching[0]['title']}' to {when} ({new_date})."
        return f"I couldn't find '{title}'."

    # --- QUERY MEMORY ---
    if intent == "query_memory":
        topic = entities.get("topic", "")
        memories = get_memories(uid, topic)
        if memories:
            msg = f"Here's what I found about '{topic}':\n\n"
            for m in memories[:5]:
                msg += f"• {m.get('content','')}\n"
            return msg
        return f"I couldn't find anything about '{topic}' in your memory."

    # --- IDEA CAPTURE ---
    if intent == "idea_capture":
        title = entities.get("title", "New idea")
        add_idea(uid, title)
        return f"💡 Idea saved:\n{title}"

    # --- MORNING BRIEF ---
    if intent == "morning_brief_request":
        await cmd_brief(update, context)
        return None

    # --- LIST TASKS ---
    if intent == "list_tasks":
        await cmd_tasks(update, context)
        return None

    # --- LIST REMINDERS ---
    if intent == "list_reminders":
        await cmd_reminders(update, context)
        return None

    # --- CHECK CALENDAR ---
    if intent == "check_calendar":
        await cmd_events(update, context)
        return None

    # --- FOCUS TIME ---
    if intent == "focus_time":
        peak = uctx.get("peak_hours", "9-12") if uctx else "9-12"
        return f"🛡️ Focus time protected for your peak hours: {peak}. No interruptions!"

    # --- STRESS CHECK ---
    if intent == "stress_check":
        pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
        return (
            "I can see you're feeling overwhelmed. Let me help:\n\n"
            f"You have {len(pending)} pending tasks.\n\n"
            "Here's what I recommend:\n"
            "1. Take a 10-minute break\n"
            "2. Focus on just ONE task next\n"
            "3. Say 'Protect my focus time' to block distractions\n\n"
            "You've got this!"
        )

    # --- SUGGESTION REQUEST ---
    if intent == "suggestion_request":
        if HAS_GENAI and GEMINI_API_KEY:
            try:
                pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
                habits = get_habits(uid, "task_completed")
                name = uctx.get("name", "there") if uctx else "there"
                ctx = f"User {name} has {len(pending)} pending tasks. "
                if pending:
                    ctx += "Tasks: " + ", ".join([t["title"] for t in pending[:5]]) + ". "
                if habits:
                    ctx += f"They usually complete tasks around {habits[0]['hour']}:00. "
                # Check if working hours are exceeded
                peak = uctx.get("peak_hours", "9-17") if uctx else "9-17"
                try:
                    end_hour = int(peak.split("-")[-1].strip().split(",")[-1].strip())
                except:
                    end_hour = 17
                now_hour = brain.now().hour
                if now_hour >= end_hour and pending:
                    ctx += f"It's now {now_hour}:00, past their peak hours ({peak}). "

                original_msg = entities.get("original_message", "What do you suggest?")
                prompt = f"Context: {ctx}\nUser asked: '{original_msg}'\nGive a warm, conversational, actionable suggestion as Mochi. Be specific and contextual. Use the user's name ({name}) if appropriate."
                client = genai.Client(api_key=GEMINI_API_KEY)
                r = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction="You are Mochi, a warm, conversational personal executive assistant. You speak like a helpful friend. NEVER say 'I am an AI assistant'. Give detailed, contextual advice."
                    )
                )
                return r.text.strip()
            except Exception as e:
                print(f"[Suggestion] Gemini error: {e}")
        pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
        if pending:
            return f"Based on your tasks, I suggest tackling '{pending[0]['title']}' first during your peak hours."
        return "You have a clear schedule! Maybe use this time for deep work or learning."

    # --- WAKE WORD ---
    if intent == "wake_word":
        return resp_text

    # --- GENERAL CHAT ---
    if intent == "general_chat":
        add_memory(uid, resp_text, category="conversation")
        return resp_text

    # --- CONFIRM ---
    if intent in ("confirm_last", "confirm_selection"):
        return resp_text

    return resp_text

async def _handle_cancel(update, context, uid, intent, entities, uctx):
    query = entities.get("query", "").lower()

    # FIX: Check if there's a pending cancel_selection action first
    pending = get_pending_actions(uid)
    if pending:
        latest = pending[0]
        data = json.loads(latest["action_data"]) if isinstance(latest["action_data"], str) else latest["action_data"]
        if data.get("pending_action") == "cancel_selection" or (isinstance(data, dict) and ("events" in data or "tasks" in data or "reminders" in data)):
            num_match = re.search(r"(\d+)", query)
            if num_match:
                selection = int(num_match.group(1))
                opts = data
                all_items = []
                for e in opts.get("events", []):
                    all_items.append(("event", e))
                for t in opts.get("tasks", []):
                    all_items.append(("task", t))
                for r in opts.get("reminders", []):
                    all_items.append(("reminder", r))
                if 1 <= selection <= len(all_items):
                    item_type, item = all_items[selection - 1]
                    if item_type == "event":
                        delete_event(uid, event_id=item["id"])
                        return f"✅ Cancelled event: {item['title']}"
                    elif item_type == "task":
                        delete_task(uid, task_id=item["id"])
                        return f"✅ Cancelled task: {item['title']}"
                    else:
                        delete_reminder(uid, reminder_id=item["id"])
                        return f"✅ Cancelled reminder: {item['title']}"
                return f"Invalid selection. Please choose a number between 1 and {len(all_items)}."

    # "cancel task 1" / "remove task 2" / "delete task 3"
    task_num_match = re.search(r"(?:cancel|remove|delete)\s+(?:task\s*)?(\d+)", query)
    if task_num_match:
        all_tasks = get_tasks(uid)
        idx = int(task_num_match.group(1)) - 1
        if 0 <= idx < len(all_tasks):
            deleted_title = all_tasks[idx]["title"]
            delete_task(uid, task_id=all_tasks[idx]["id"])
            # Show updated list
            remaining = get_tasks(uid)
            text = f"✅ Cancelled task {task_num_match.group(1)}: {deleted_title}\n\n"
            if remaining:
                text += "Updated task list:\n"
                for i, t in enumerate(remaining, 1):
                    status = "[x]" if t.get("status") == "completed" else "[ ]"
                    text += f"{status} {i}. {t['title']} (Due: {t.get('due_date','No date')}, {t.get('priority','medium').upper()})\n"
            else:
                text += "Your task list is now empty."
            return text
        return f"Task {task_num_match.group(1)} doesn't exist."

    # "clear all tasks" / "clear the whole task list" / "delete all tasks"
    if any(p in query for p in ["all tasks", "whole task list", "clear task", "delete all task", "clear everything"]):
        delete_all_tasks(uid)
        return "✅ All tasks cleared."

    # FIX: "cancel all events" / "clear all events" / "delete all events"
    if any(p in query for p in ["all events", "clear all event", "delete all event", "clear event list"]):
        events = get_events(uid)
        for e in events:
            delete_event(uid, event_id=e["id"])
        return f"✅ Cleared {len(events)} events from your calendar."

    events = get_events(uid)
    tasks = get_tasks(uid)
    rems = get_user_reminders(uid)

    me = [e for e in events if any(w in e.get("title","").lower() for w in query.split())]
    mt = [t for t in tasks if any(w in t.get("title","").lower() for w in query.split())]
    mr = [r for r in rems if any(w in r.get("title","").lower() for w in query.split())]

    total = len(me) + len(mt) + len(mr)
    if total == 0:
        return "I couldn't find anything matching that to cancel. Could you be more specific?"
    if total == 1:
        if me:
            delete_event(uid, event_id=me[0]["id"])
            return f"✅ Cancelled event: {me[0]['title']}"
        if mt:
            delete_task(uid, task_id=mt[0]["id"])
            return f"✅ Cancelled task: {mt[0]['title']}"
        delete_reminder(uid, reminder_id=mr[0]["id"])
        return f"✅ Cancelled reminder: {mr[0]['title']}"

    text = "I found multiple matches. Which one? (reply with the number)\n\n"
    idx = 1
    opts = {"events": [], "tasks": [], "reminders": []}
    for e in me[:3]:
        text += f"{idx}. 📅 Event: {e['title']} ({e.get('date','')} {e.get('time','')})\n"
        opts["events"].append(e)
        idx += 1
    for t in mt[:3]:
        text += f"{idx}. ✅ Task: {t['title']}\n"
        opts["tasks"].append(t)
        idx += 1
    for r in mr[:3]:
        text += f"{idx}. ⏰ Reminder: {r['title']}\n"
        opts["reminders"].append(r)
        idx += 1

    add_pending_action(uid, "cancel_selection", opts, question="Which one to cancel?")
    return text

async def _handle_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = update.message.text
    state = user_setup[uid]
    step = state["step"]
    key, _ = SETUP_QUESTIONS[step]
    state["data"][key] = text
    state["step"] += 1

    if state["step"] < len(SETUP_QUESTIONS):
        _, next_q = SETUP_QUESTIONS[state["step"]]
        await update.message.reply_text(next_q)
    else:
        data = state["data"]
        voice = data.get("voice", "female").lower()
        if voice not in ["male", "female"]:
            voice = "female"
        add_user(
            user_id=uid,
            name=data.get("name", "User"),
            role=data.get("role", ""),
            goals=data.get("goals", ""),
            timezone=data.get("timezone", "UTC"),
            peak_hours=data.get("peak_hours", "9-17"),
            briefing_time=data.get("briefing_time", "08:00"),
            voice=voice
        )
        del user_setup[uid]
        name = data.get("name", "User")
        await update.message.reply_text(
            f"Profile complete! Welcome {name}!\n\n"
            "I'm Mochi, your executive assistant.\n\n"
            "Try: 'Meeting tomorrow at 2pm' or 'Remind me in 5 minutes to prepare lunch'."
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid in user_setup:
        await update.message.reply_text("Please complete setup with text first.")
        return

    voice_file = await update.message.voice.get_file()
    tmp_dir = tempfile.gettempdir()
    voice_path = os.path.join(tmp_dir, f"voice_{uid}.ogg")
    await voice_file.download_to_drive(voice_path)

    try:
        from core.transcriber import transcribe_voice
        text = transcribe_voice(voice_path)
        if not text:
            await update.message.reply_text("Sorry, I couldn't understand that. Could you try again?")
            return
        await update.message.reply_text(f"Heard: {text}")

        user = get_user(uid)
        uctx = _user_ctx(user)
        chat_hist = get_chat_history(uid, limit=10)
        pending = get_pending_actions(uid)
        track_habit(uid, "message_received", hour=brain.now().hour)

        result = brain.process(text, uctx, chat_hist, pending, is_voice=True)
        intent = result.get("intent", "general_chat")
        entities = result.get("entities", {})
        resp_text = result.get("response_text", "I'm Mochi. How can I help?")

        add_chat_message(uid, "user", text)
        action_resp = await _execute_intent(update, context, uid, intent, entities, resp_text, uctx)
        if action_resp:
            add_chat_message(uid, "assistant", action_resp, intent=intent, entities=entities)
            await _reply(update, action_resp, uctx)
    except Exception as e:
        print(f"[Voice] Error: {e}")
        traceback.print_exc()
        await update.message.reply_text("Sorry, I had trouble with that voice message. Please try again.")
    finally:
        if os.path.exists(voice_path):
            os.remove(voice_path)

# ------------------------------------------------------------------
# BACKGROUND LOOPS
# ------------------------------------------------------------------
async def reminder_loop(app):
    while True:
        try:
            now = brain.now().strftime("%Y-%m-%dT%H:%M:%S")
            due = get_due_reminders(now)
            for rem in due:
                try:
                    await app.bot.send_message(chat_id=rem["user_id"], text=f"⏰ Reminder: {rem['title']}")
                    mark_reminder_sent(rem["id"])
                except Exception as e:
                    print(f"[Reminder] Failed to send to {rem['user_id']}: {e}")
        except Exception as e:
            print(f"[Reminder Loop] Error: {e}")
        await asyncio.sleep(15)

async def briefing_loop(app):
    while True:
        try:
            now = brain.now()
            today = brain.today().strftime("%Y-%m-%d")
            users = get_all_users()
            for user in users:
                if not user.get("briefing_time"):
                    continue
                try:
                    tz = pytz.timezone(user.get("timezone", "UTC"))
                except:
                    tz = pytz.UTC
                local_now = now.astimezone(tz)
                if local_now.strftime("%H:%M") == user["briefing_time"]:
                    if not was_briefing_sent_today(user["user_id"], today):
                        try:
                            await send_briefing(app, user)
                        except Exception as e:
                            print(f"[Briefing] Failed for {user['user_id']}: {e}")
        except Exception as e:
            print(f"[Briefing Loop] Error: {e}")
        await asyncio.sleep(30)

async def send_briefing(app, user):
    uid = user["user_id"]
    name = user.get("name", "there")
    today = brain.today().strftime("%Y-%m-%d")
    now_time = brain.now().strftime("%H:%M")
    events = get_events(uid, date=today)
    past = [e for e in events if e.get("time","00:00") < now_time]
    future = [e for e in events if e.get("time","00:00") >= now_time]
    pending = [t for t in get_tasks(uid) if t.get("status") != "completed"]
    text = f"Good morning {name}! Here's your briefing:\n\n"
    if future:
        text += f"Upcoming Events ({len(future)}):\n"
        for e in future:
            text += f"• {e.get('time','??:??')} — {e['title']}\n"
        text += "\n"
    else:
        text += "Upcoming Events: Nothing ahead!\n\n"
    if past:
        text += f"Completed Today ({len(past)}):\n"
        for e in past:
            text += f"• ~~{e.get('time','??:??')} — {e['title']}~~\n"
        text += "\n"
    if pending:
        text += f"Pending Tasks ({len(pending)}):\n"
        for t in pending[:5]:
            text += f"• {t['title']}\n"
    else:
        text += "Pending Tasks: All clear!\n"
    await app.bot.send_message(chat_id=uid, text=text)
    mark_briefing_sent(uid, today)

async def eod_check_loop(app):
    while True:
        try:
            now = brain.now()
            today = brain.today().strftime("%Y-%m-%d")
            users = get_all_users()
            for user in users:
                try:
                    tz = pytz.timezone(user.get("timezone", "UTC"))
                except:
                    tz = pytz.UTC
                local_now = now.astimezone(tz)
                peak = user.get("peak_hours", "9-17")
                end_hour = 17
                for block in peak.split(","):
                    if "-" in block:
                        _, end = block.split("-")
                        end_hour = int(end.strip())
                if local_now.hour == end_hour and local_now.minute == 0:
                    if not was_eod_sent_today(user["user_id"], today):
                        pending = [t for t in get_tasks(user["user_id"]) if t.get("status") != "completed"]
                        if pending:
                            # Smart suggestion: suggest moving based on priority
                            high_priority = [t for t in pending if t.get("priority") == "high"]
                            low_priority = [t for t in pending if t.get("priority") == "low"]
                            msg = f"Hey {user.get('name', 'there')}! Your productive hours are ending ({peak}).\n\n"
                            msg += f"You still have {len(pending)} pending tasks:\n"
                            for t in pending[:5]:
                                pri = t.get("priority", "medium").upper()
                                msg += f"• [{pri}] {t['title']}\n"
                            msg += "\n"
                            if low_priority:
                                msg += f"💡 I suggest moving {len(low_priority)} low-priority task(s) to tomorrow.\n"
                            if high_priority:
                                msg += f"🔥 But keep {len(high_priority)} high-priority task(s) for tonight if you can.\n"
                            msg += "\nWould you like me to move the low-priority ones to tomorrow? (Reply yes or no)"
                            await app.bot.send_message(chat_id=user["user_id"], text=msg)
                            add_pending_action(user["user_id"], "eod_move",
                                               {"pending_action": "eod_move", "tasks": pending},
                                               question="Move low-priority tasks to tomorrow?")
                            mark_eod_sent(user["user_id"], today)
        except Exception as e:
            print(f"[EOD Loop] Error: {e}")
        await asyncio.sleep(60)

# ------------------------------------------------------------------
# TASK DUE DATE REMINDER LOOP
# ------------------------------------------------------------------
async def task_due_reminder_loop(app):
    """Check for tasks due today or tomorrow and send reminders."""
    while True:
        try:
            now = brain.now()
            today = brain.today().strftime("%Y-%m-%d")
            tomorrow = (brain.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            users = get_all_users()
            for user in users:
                uid = user["user_id"]
                tasks = get_tasks(uid, status="pending")
                for t in tasks:
                    due = t.get("due_date", "")
                    if not due:
                        continue
                    # Remind if due today and it's morning (9am)
                    if due == today and now.hour == 9 and now.minute < 5:
                        try:
                            await app.bot.send_message(
                                chat_id=uid,
                                text=f"⏰ Reminder: '{t['title']}' is due TODAY! (Priority: {t.get('priority','medium').upper()}) Want me to help you prioritize your day?"
                            )
                        except Exception as e:
                            print(f"[Task Due Reminder] Failed: {e}")
                    # Remind if due tomorrow and it's evening (8pm)
                    if due == tomorrow and now.hour == 20 and now.minute < 5:
                        try:
                            await app.bot.send_message(
                                chat_id=uid,
                                text=f"📅 Heads up: '{t['title']}' is due TOMORROW. (Priority: {t.get('priority','medium').upper()}) Would you like to block some time for it?"
                            )
                        except Exception as e:
                            print(f"[Task Due Reminder] Failed: {e}")
        except Exception as e:
            print(f"[Task Due Loop] Error: {e}")
        await asyncio.sleep(300)  # Check every 5 minutes

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
async def _run():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("events", cmd_events))
    app.add_handler(CommandHandler("reminders", cmd_reminders))
    app.add_handler(CommandHandler("brief", cmd_brief))
    app.add_handler(CommandHandler("auth_calendar", cmd_auth_calendar))
    app.add_handler(CommandHandler("calendar_code", cmd_calendar_code))
    app.add_handler(CommandHandler("sync_tasks", cmd_sync_tasks))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    await app.initialize()
    await app.start()

    asyncio.create_task(reminder_loop(app))
    asyncio.create_task(briefing_loop(app))
    asyncio.create_task(eod_check_loop(app))
    asyncio.create_task(task_due_reminder_loop(app))

    await app.updater.start_polling()
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def main():
    init_db()
    print("=" * 50)
    print("Mochi is running! Using real-time dates.")
    print("Send /start in Telegram to begin.")
    print("=" * 50)
    asyncio.run(_run())

if __name__ == "__main__":
    main()
