"""Mochi's Brain — NLU, intent detection, memory, decision engine."""
import os
import re
import json
from datetime import datetime, timedelta, timezone
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_NUM = {d: i for i, d in enumerate(DAYS)}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
          "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}

class Brain:
    def __init__(self):
        self.client = None
        self.model = GEMINI_MODEL or "gemini-2.0-flash"
        if GEMINI_API_KEY and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                print("[Brain] Gemini ready")
            except Exception as e:
                print(f"[Brain] Gemini init failed: {e}")
        else:
            print("[Brain] Running rule-based only")

    def today(self):
        return datetime.now(timezone.utc).date()

    def now(self):
        return datetime.now(timezone.utc)
        # ------------------------------------------------------------------
    # VOICE PREPROCESSING — normalize transcription artifacts
    # ------------------------------------------------------------------
    def _preprocess_voice(self, msg):
        """Normalize common voice transcription artifacts before parsing."""
        msg = msg.strip()
        msg_lower = msg.lower()
        
        # Fix: "Need to..." -> "I need to..." (Deepgram often drops leading "I")
        if msg_lower.startswith("need to ") and not msg_lower.startswith("i need to "):
            msg = "I " + msg
        if msg_lower.startswith("have to ") and not msg_lower.startswith("i have to "):
            msg = "I " + msg
        if msg_lower.startswith("want to ") and not msg_lower.startswith("i want to "):
            msg = "I " + msg
        if msg_lower.startswith("should ") and not msg_lower.startswith("i should "):
            msg = "I " + msg
        if msg_lower.startswith("must ") and not msg_lower.startswith("i must "):
            msg = "I " + msg
            
        # Fix: "Gotta" -> "I have to"
        if msg_lower.startswith("gotta "):
            msg = "I have to " + msg[6:]
        if msg_lower.startswith("gonna "):
            msg = "I am going to " + msg[6:]
            
        # Fix: "Wanna" -> "I want to"
        if msg_lower.startswith("wanna "):
            msg = "I want to " + msg[6:]
            
        # Fix: "Lemme" -> "Let me"
        if msg_lower.startswith("lemme "):
            msg = "Let me " + msg[6:]
            
        # Fix: "Imma" -> "I am going to"
        if msg_lower.startswith("imma "):
            msg = "I am going to " + msg[5:]
            
        # Fix: "Give me" -> "Show me" for list queries
        if msg_lower.startswith("give me my "):
            msg = "show my " + msg[11:]
        if msg_lower.startswith("give me the "):
            msg = "show the " + msg[12:]
        if msg_lower.startswith("give me "):
            msg = "show " + msg[8:]
            
        # Fix: "What about today/tomorrow" -> calendar query
        if msg_lower.startswith("what about ") and any(w in msg_lower for w in ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            msg = "what do I have " + msg[11:]
            
        # Fix: "Do we have" -> "what do I have" (from your screenshots)
        if msg_lower.startswith("do we have "):
            msg = "what do I have " + msg[11:]
        if msg_lower.startswith("do we have anything "):
            msg = "what do I have " + msg[19:]
        if msg_lower.startswith("what do we have "):
            msg = "what do I have " + msg[16:]
            
        # Fix: "Any events/tasks/reminders" -> "what events/tasks/reminders"
        if msg_lower.startswith("any events"):
            msg = "what events" + msg[10:]
        if msg_lower.startswith("any tasks"):
            msg = "what tasks" + msg[9:]
        if msg_lower.startswith("any reminders"):
            msg = "what reminders" + msg[13:]
            
        # Fix: "Schedule me" -> "Schedule a"
        if msg_lower.startswith("schedule me "):
            msg = "schedule a " + msg[12:]
            
        # Fix: "Take shower" -> "take a shower"
        msg = re.sub(r"\btake shower\b", "take a shower", msg, flags=re.IGNORECASE)
        
        # Fix: common voice contractions
        msg = re.sub(r"\bim\b", "I'm", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bdont\b", "don't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bwont\b", "won't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bcant\b", "can't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bdidnt\b", "didn't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bwasnt\b", "wasn't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bisnt\b", "isn't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\barent\b", "aren't", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\btheres\b", "there's", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bthats\b", "that's", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bwhats\b", "what's", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bheres\b", "here's", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bhows\b", "how's", msg, flags=re.IGNORECASE)
        
        # Fix: "Could ya" / "Can ya" -> "Could you" / "Can you"
        msg = re.sub(r"\b(could|can|will|would)\s+ya\b", r"\1 you", msg, flags=re.IGNORECASE)
        
        # Fix: "Kinda", "Sorta", "Dunno", "Gimme"
        msg = re.sub(r"\bkinda\b", "kind of", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bsorta\b", "sort of", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bdunno\b", "don't know", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\bgimme\b", "give me", msg, flags=re.IGNORECASE)
        
        # Fix: "Tell me" -> "show me" for list queries
        if msg_lower.startswith("tell me my "):
            msg = "show my " + msg[11:]
        if msg_lower.startswith("tell me the "):
            msg = "show the " + msg[12:]
            
        return msg
    
    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------
    def process(self, message, user_context=None, chat_history=None, pending_actions=None, is_voice=False):
        # VOICE FIX: Preprocess voice transcriptions before parsing
        if is_voice:
            message = self._preprocess_voice(message)
        msg = message.strip()
        msg_lower = msg.lower()
        name = user_context.get("name", "") if user_context else ""
        name_str = f" {name}" if name else ""

        # 1. Pending actions — only if message looks like a short follow-up
        if pending_actions:
            r = self._handle_pending(msg, msg_lower, pending_actions, user_context)
            if r:
                return r

        # 2. Cancellations / deletions
        r = self._detect_cancellation(msg, msg_lower)
        if r:
            return r

        # 3. Task operations by number or keyword
        r = self._detect_task_ops(msg, msg_lower, user_context)
        if r:
            return r

        # 4. Identity
        if self._is_identity_query(msg_lower):
            return self._identity_response(user_context)

        # 5. Date query
        if self._is_date_query(msg_lower):
            return self._date_response(user_context)

        # 6. Context references
        r = self._resolve_context(msg, msg_lower, chat_history)
        if r:
            return r

        # 7. Morning brief / schedule brief
        if self._is_morning_brief(msg_lower):
            return {"intent": "morning_brief_request", "entities": {},
                    "response_text": f"Let me get your morning briefing ready{name_str}."}
        if self._is_schedule_brief(msg_lower):
            bt = user_context.get("briefing_time", "08:00") if user_context else "08:00"
            return {"intent": "schedule_briefing", "entities": {"briefing_time": bt},
                    "response_text": f"Got it{name_str}! I'll send your morning briefing at {bt} every day. Sleep well!"}

        # 8. Check queries (calendar, memory, tomorrow, today, yesterday, tasks, events)
        r = self._detect_check_queries(msg, msg_lower, user_context)
        if r:
            return r

        # 9. List commands
        r = self._detect_list_queries(msg, msg_lower, user_context)
        if r:
            return r

        # 10. Suggestion request — use Gemini for free-form answers
        if self._is_suggestion_request(msg_lower):
            return {"intent": "suggestion_request", "entities": {"original_message": msg},
                    "response_text": "Let me think about what would work best for you..."}

        # 11. Reminder
        r = self._try_reminder(msg, msg_lower, user_context)
        if r:
            return r

        # 12. Mixed sentences
        r = self._try_mixed(msg, msg_lower, user_context)
        if r:
            return r

        # 13. Event
        r = self._try_event(msg, msg_lower, user_context)
        if r:
            return r

        # 14. Task (with multi-task support)
        r = self._try_task(msg, msg_lower, user_context)
        if r:
            return r
                # VOICE FIX: Try voice-specific task patterns (imperative/command forms)
        r = self._try_voice_task(msg, msg_lower, user_context)
        if r:
            return r
        
        # 15. Idea
        r = self._try_idea(msg, msg_lower, user_context)
        if r:
            return r

        # 16. Focus / stress
        if self._is_focus_time(msg_lower):
            peak = user_context.get("peak_hours", "9-12") if user_context else "9-12"
            return {"intent": "focus_time", "entities": {},
                    "response_text": f"I'll protect your focus time during your peak hours ({peak})."}
        if self._is_stress_check(msg_lower):
            return {"intent": "stress_check", "entities": {},
                    "response_text": "I can see you're feeling overwhelmed. Let me help you prioritize."}

        # 17. Gemini fallback — ALWAYS try Gemini before generic fallback
        if self.client:
            try:
                r = self._gemini_process(msg, user_context, chat_history)
                if r and r.get("intent"):
                    resp = r.get("response_text", "")
                    if name and "I'm Mochi" in resp and name not in resp:
                        resp = resp.replace("I'm Mochi", f"I'm Mochi, {name}'s assistant")
                        r["response_text"] = resp
                    return r
            except Exception as e:
                print(f"[Brain] Gemini error: {e}")

        # 18. Wake word
        if self._is_wake_word(msg_lower):
            return {"intent": "wake_word", "entities": {},
                    "response_text": f"Yes{name_str}, I'm Mochi. What can I do for you?"}

        # 19. Fallback — NEVER generic, always contextual
        return {"intent": "general_chat", "entities": {},
                "response_text": f"Hey{name_str}! I'm Mochi, your executive assistant. I can schedule events, set reminders, manage tasks, or check your calendar. What would you like to do?"}

    # ------------------------------------------------------------------
    # PENDING ACTION HANDLER
    # ------------------------------------------------------------------
    def _handle_pending(self, msg, msg_lower, pending_actions, user_context):
        new_cmds = ["schedule", "check", "what", "cancel", "add", "create", "list", "show",
                    "mark", "make", "move", "delete", "remind me", "i need to", "i have to",
                    "good morning", "who am i", "my tasks", "my events", "what is", "what are",
                    "how many", "did you", "can you", "will you", "do you", "where", "when is",
                    "who is", "tell me", "give me", "i want to", "i'd like to", "complete",
                    "finish", "remove", "clear", "postpone", "rename", "edit", "suggest",
                    "recommend", "help me", "what about", "what do we", "show me",                    # VOICE FIX: add voice command starters
                    "need to", "have to", "want to", "should", "must", "work out", "workout",
                    "exercise", "take a", "finish", "complete", "prepare", "call", "buy",
                    "get", "read", "study", "send", "write", "clean", "pay", "handle",
                    "start", "begin", "book", "reserve", "gotta", "wanna", "gonna"]
        if len(msg) > 10 and any(c in msg_lower for c in new_cmds):
            return None

        latest = pending_actions[0]
        data = json.loads(latest["action_data"]) if isinstance(latest["action_data"], str) else latest["action_data"]

        if data.get("pending_action") == "set_reminder":
            dt_info = self._extract_time(msg_lower, user_context)
            has_time = dt_info[2]
            target = dt_info[3]
            if has_time and target:
                return {"intent": "set_reminder",
                        "entities": {"title": data["title"], "remind_at": target.strftime("%Y-%m-%dT%H:%M:%S")},
                        "response_text": f"Reminder set! I'll remind you about '{data['title']}' at {target.strftime('%H:%M UTC')}."}
            if len(msg) < 30 and target:
                return {"intent": "set_reminder",
                        "entities": {"title": data["title"], "remind_at": target.strftime("%Y-%m-%dT%H:%M:%S")},
                        "response_text": f"Reminder set! I'll remind you about '{data['title']}' at {target.strftime('%H:%M UTC')}."}
            return {"intent": "ask_followup",
                    "entities": data,
                    "response_text": "I need a specific time. Try: 'in 10 minutes', 'at 5pm', or 'tomorrow at 9am'."}

        if data.get("pending_action") == "create_event":
            dt_info = self._extract_time(msg_lower, user_context)
            has_time = dt_info[2]
            target = dt_info[3]
            if has_time and target:
                return {"intent": "create_event",
                        "entities": {"title": data["title"], "date": dt_info[0], "time": dt_info[1], "duration": 60, "people": data.get("people", [])},
                        "response_text": f"Event scheduled: {data['title']} on {dt_info[0]} at {dt_info[1]}."}
            if len(msg) < 30 and target:
                return {"intent": "create_event",
                        "entities": {"title": data["title"], "date": dt_info[0], "time": dt_info[1], "duration": 60, "people": data.get("people", [])},
                        "response_text": f"Event scheduled: {data['title']} on {dt_info[0]} at {dt_info[1]}."}
            return {"intent": "ask_followup",
                    "entities": data,
                    "response_text": "What time should I schedule it for? (e.g., 'at 3pm')"}

        if data.get("pending_action") == "move_task":
            if "tomorrow" in msg_lower:
                return {"intent": "move_task",
                        "entities": {"title": data["title"], "when": "tomorrow"},
                        "response_text": f"Moved '{data['title']}' to tomorrow."}
            if "today" in msg_lower:
                return {"intent": "move_task",
                        "entities": {"title": data["title"], "when": "today"},
                        "response_text": f"Kept '{data['title']}' for today."}
            if len(msg) < 15:
                return {"intent": "ask_followup",
                        "entities": data,
                        "response_text": "Would you like to move it to tomorrow or keep it for today?"}
            return None

        if data.get("pending_action") == "cancel_selection":
            num_match = re.search(r"(\d+)", msg_lower)
            if num_match:
                return {"intent": "cancel_by_number",
                        "entities": {"selection": int(num_match.group(1)), "options": data},
                        "response_text": "Cancelling that now."}
            if len(msg) < 15:
                return {"intent": "ask_followup",
                        "entities": data,
                        "response_text": "Please reply with the number of the item you want to cancel."}
            return None

        if len(msg) < 15:
            if re.search(r"^yes\b|^yeah\b|^sure\b|^ok\b|^okay\b", msg_lower):
                return {"intent": "confirm_last", "entities": {},
                        "response_text": "Great! I'll proceed with that."}
            if re.search(r"^no\b|^nope\b|^cancel\b", msg_lower):
                return {"intent": "general_chat", "entities": {},
                        "response_text": "No problem! Let me know if you need anything else."}

        return None

    # ------------------------------------------------------------------
    # TASK OPERATIONS
    # ------------------------------------------------------------------
    def _detect_task_ops(self, msg, msg_lower, user_context):
        name = user_context.get("name", "") if user_context else ""
        name_str = f" {name}" if name else ""

        # Mark all / complete all / finish all / clear all
        if re.search(r"(?:mark|complete|finish)\s+(?:them\s+)?all\s+(?:as\s+)?(?:done|completed)?", msg_lower):
            return {"intent": "mark_all_done", "entities": {},
                    "response_text": f"Marking all tasks as completed{name_str}."}
        if re.search(r"clear\s+(?:all\s+)?(?:completed\s+)?tasks", msg_lower) or \
           re.search(r"clear\s+the\s+whole\s+task\s+list", msg_lower) or \
           re.search(r"delete\s+all\s+tasks", msg_lower) or \
           re.search(r"remove\s+all\s+tasks", msg_lower):
            return {"intent": "clear_all_tasks", "entities": {},
                    "response_text": f"Clearing all tasks{name_str}."}

        # Remove / delete / cancel task by number
        m = re.search(r"(?:remove|delete|cancel)\s+(?:task\s*)?(\d+)", msg_lower)
        if m:
            return {"intent": "delete_task_by_number", "entities": {"number": int(m.group(1))},
                    "response_text": f"Removing task {m.group(1)}."}

        # Mark / complete / finish task by number
        m = re.search(r"(?:mark|complete|finish)\s+(?:task\s*)?(\d+(?:\s+and\s+\d+)*)\s*(?:as\s+)?(?:done|completed)?", msg_lower)
        if m:
            nums = re.findall(r"\d+", m.group(1))
            return {"intent": "mark_task_by_number", "entities": {"numbers": [int(n) for n in nums]},
                    "response_text": f"Marking task(s) {', '.join(nums)} as completed."}

        # Rename task by number
        m = re.search(r"(?:rename|change|edit)\s+(?:task\s*)?(\d+)\s+(?:to\s+)?(.+)", msg_lower)
        if m:
            return {"intent": "rename_task", "entities": {"number": int(m.group(1)), "new_title": m.group(2).strip()},
                    "response_text": f"Renaming task {m.group(1)}."}

        # Move / postpone task by number
        m = re.search(r"(?:move|postpone)\s+(?:task\s*)?(\d+)\s+(?:to|until|for)\s+(.+)", msg_lower)
        if m:
            when = m.group(2).strip()
            dt_info = self._parse_date_word(when)
            if dt_info:
                return {"intent": "move_task_by_number",
                        "entities": {"number": int(m.group(1)), "date": dt_info[0]},
                        "response_text": f"Moving task {m.group(1)} to {dt_info[0]}."}
            return {"intent": "move_task_by_number",
                    "entities": {"number": int(m.group(1)), "when": when},
                    "response_text": f"Moving task {m.group(1)} to {when}."}

        # FIX: Change priority by number - handle "change the priority of task 2 to high"
        m = re.search(r"(?:make|set|change)\s+(?:the\s+priority\s+of\s+)?(?:task\s*)?(\d+)\s+(?:to\s+)?(high|low|medium|urgent)", msg_lower)
        if not m:
            m = re.search(r"(?:make|set|change)\s+(?:task\s*)?(\d+)\s+(?:priority\s+)?(?:to\s+)?(high|low|medium|urgent)", msg_lower)
        if not m:
            m = re.search(r"(?:make|set|change)\s+(?:priority\s+of\s+)?(?:task\s*)?(\d+)\s+(?:to\s+)?(high|low|medium|urgent)", msg_lower)
        if m:
            return {"intent": "change_priority_by_number",
                    "entities": {"number": int(m.group(1)), "priority": m.group(2)},
                    "response_text": f"Changing task {m.group(1)} priority to {m.group(2)}."}

        # "second task", "the first one", "the last task"
        m = re.search(r"(?:the\s+)?(first|second|third|last)\s+task", msg_lower)
        if m:
            pos = m.group(1)
            return {"intent": "refer_task_by_position", "entities": {"position": pos},
                    "response_text": f"Referring to the {pos} task."}

        # FIX: "i did the workout mark it as done" / "i finished the report" / "mark workout as done"
        m = re.search(r"(?:i\s+did|i\s+finished|i\s+completed|i\s+have\s+done)\s+(.+?)(?:\s+mark\s+it\s+as\s+done|\s+mark\s+as\s+done|\s+done)?", msg_lower)
        if m:
            title = m.group(1).strip()
            return {"intent": "mark_task_done", "entities": {"title": title},
                    "response_text": f"Marking '{title}' as done."}

        # "mark [task name] as done"
        m = re.search(r"(?:mark)\s+(.+?)\s+(?:as\s+)?(?:done|completed|finished)", msg_lower)
        if m and not re.search(r"(?:mark|complete|finish)\s+(?:task\s*)?\d+", msg_lower):
            title = m.group(1).strip()
            return {"intent": "mark_task_done", "entities": {"title": title},
                    "response_text": f"Marking '{title}' as done."}

        # FIX: Edit task by title - "change [task] to [new title]" / "rename [task] to [new title]"
        m = re.search(r"(?:change|rename|edit)\s+(.+?)\s+(?:to|into)\s+(.+)", msg_lower)
        if m:
            old_title = m.group(1).strip()
            new_title = m.group(2).strip()
            return {"intent": "edit_task_title", "entities": {"old_title": old_title, "new_title": new_title},
                    "response_text": f"Renaming '{old_title}' to '{new_title}'."}

        # FIX: Change task date by title - "move [task] to [date]"
        m = re.search(r"(?:move|change|update)\s+(.+?)\s+(?:date|to|for)\s+(.+)", msg_lower)
        if m:
            title = m.group(1).strip()
            new_date = m.group(2).strip()
            if any(d in new_date for d in ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next", "week"]):
                return {"intent": "edit_task_date", "entities": {"title": title, "new_date": new_date},
                        "response_text": f"Moving '{title}' to {new_date}."}

        # FIX: Edit event by title - "change event [title] to [new title]"
        m = re.search(r"(?:change|rename|edit)\s+(?:event\s+)?(.+?)\s+(?:to|into)\s+(.+)", msg_lower)
        if m and any(w in msg_lower for w in ["event", "meeting", "appointment"]):
            old_title = m.group(1).strip()
            new_title = m.group(2).strip()
            return {"intent": "edit_event", "entities": {"old_title": old_title, "new_title": new_title},
                    "response_text": f"Updating event '{old_title}' to '{new_title}'."}

        # FIX: Change working hours for today - "today I work from X to Y"
        m = re.search(r"(?:today|tomorrow)\s+i\s+(?:decided\s+to\s+)?(?:work|working)\s+(?:from\s+)?(\d{1,2})(?::\d{2})?\s*(?:am|pm)?\s+(?:to|until|till)\s+(\d{1,2})(?::\d{2})?\s*(?:am|pm)?", msg_lower)
        if m:
            day = "today" if "today" in msg_lower else "tomorrow"
            return {"intent": "set_working_hours", "entities": {"day": day, "start": m.group(1), "end": m.group(2)},
                    "response_text": f"Got it! I'll update your working hours for {day} to {m.group(1)}-{m.group(2)}."}

        return None

    # ------------------------------------------------------------------
    # CANCELLATION
    # ------------------------------------------------------------------
    def _detect_cancellation(self, msg, msg_lower):
        cancel_starters = ["cancel", "delete", "remove", "drop", "clear", "undo"]
        if not any(msg_lower.startswith(s) for s in cancel_starters):
            return None

        if any(w in msg_lower for w in ["event", "meeting", "appointment", "dinner", "lunch"]):
            return {"intent": "cancel_event", "entities": {"query": msg},
                    "response_text": "Let me find and cancel that event."}
        if any(w in msg_lower for w in ["task", "todo", "reminder"]):
            return {"intent": "cancel_task", "entities": {"query": msg},
                    "response_text": "Let me find and cancel that."}
        return {"intent": "cancel_search", "entities": {"query": msg},
                "response_text": "Let me search for what you want to cancel."}

    # ------------------------------------------------------------------
    # CHECK QUERIES (unified, flexible)
    # ------------------------------------------------------------------
    def _detect_check_queries(self, msg, msg_lower, user_context):
        name = user_context.get("name", "") if user_context else ""
        name_str = f" {name}" if name else ""

        # Memory queries
        if any(p in msg_lower for p in ["did you save", "did you remember", "what did i tell you",
               "what did i say", "did you plan", "did you add", "did you write down", "save what i told you"]):
            return {"intent": "query_memory", "entities": {"topic": "recent"},
                    "response_text": f"Let me check what I saved from our conversations{name_str}."}

        # Tomorrow queries
        tomorrow = (self.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        if any(p in msg_lower for p in ["did you plan tomorrow", "what about tomorrow",
               "tomorrow's plan", "tomorrow plan", "what do i have tomorrow",
               "whats tomorrow", "what's tomorrow", "tomorrow schedule", "tomorrow events",
               "calendar for tomorrow", "in my calendar for tomorrow", "what is in my calendar for tomorrow",
               "what's in my calendar for tomorrow", "what i have for tomorrow",
               "what's for tomorrow", "whats for tomorrow", "what about tomorrow",
               "what i have tomorrow", "what do i have for tomorrow", "tasks for tomorrow",
               "my tasks for tomorrow", "events for tomorrow", "anything for tomorrow",
               "do i have anything tomorrow", "what's happening tomorrow", "show me tomorrow"]):
            return {"intent": "check_calendar_date", "entities": {"date": tomorrow},
                    "response_text": f"Let me check what you have planned for tomorrow{name_str}."}

        # Today queries — VERY BROAD
        today = self.today().strftime("%Y-%m-%d")
        today_patterns = [
            "what about today", "what do i have today", "what's today", "whats today",
            "what is today", "calendar for today", "in my calendar for today",
            "what is in my calendar for today", "what's in my calendar for today",
            "what i have for today", "what do i have for today", "do we have sth for today",
            "do we have something for today", "what do we have today",
            "what do we have for today", "check my calendar for today",
            "show me today", "my calendar for today", "events for today",
            "tasks for today", "anything for today", "what's happening today",
            "what do i have scheduled today", "what am i doing today",
            "what's on my calendar today", "whats on my calendar today",
            "what i have scheduled today", "show me my schedule today",
            "what do we have for the day", "what do i have for the day",
            "what is planned for today", "what's planned for today",
            "what do i need to do today", "what should i do today",
            "do i have anything today", "anything on today", "my day today",
            "how does my day look", "how's my day looking", "what's my day like",
            "what is my day like", "what do i have on today", "what about my day",
            "what's on today", "whats on today",
            "what do i have on", "what's going on today", "whats going on today",
            "how's today looking", "how is today looking",
            "what's my schedule", "whats my schedule",
            "what do i got today", "what do i got going on",
            "what's happening today", "whats happening today",
            "show me what i got", "what i got today",
            "do i have plans today", "any plans today",
            "what's the plan today", "whats the plan today",
        ]
        if any(p in msg_lower for p in today_patterns):
            return {"intent": "check_calendar_date", "entities": {"date": today},
                    "response_text": f"Let me check what you have for today{name_str}."}

        # Yesterday queries
        yesterday = (self.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        if any(p in msg_lower for p in ["what did i do yesterday", "yesterday's events",
               "yesterday schedule", "did i finish yesterday", "what happened yesterday",
               "calendar for yesterday", "in my calendar for yesterday", "tasks for yesterday",
               "events for yesterday", "what about yesterday", "what did i have yesterday"]):
            return {"intent": "check_calendar_date", "entities": {"date": yesterday},
                    "response_text": f"Let me check what you did yesterday{name_str}."}

        # Specific date queries: "in 2 days", "in one week", "next week", etc.
        m = re.search(r"(?:what do i have|what's|whats|what about|check|show me|tasks for|events for).*?(?:in|after)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months|year|years)", msg_lower)
        if m:
            target = self._parse_relative_date(m.group(1), m.group(2))
            if target:
                return {"intent": "check_calendar_date", "entities": {"date": target.strftime("%Y-%m-%d")},
                        "response_text": f"Let me check what you have for {target.strftime('%A, %B %d')}{name_str}."}

        # Day of week queries: "what about wednesday", "tasks for friday"
        m = re.search(r"(?:what about|what do i have|tasks for|events for|check|show me).*?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", msg_lower)
        if m:
            target = self._date_from_weekday(m.group(1), next_week=False)
            if target:
                return {"intent": "check_calendar_date", "entities": {"date": target.strftime("%Y-%m-%d")},
                        "response_text": f"Let me check what you have for {m.group(1).title()}{name_str}."}

        # Remaining today
        if any(p in msg_lower for p in ["what do i still have", "what do i have left",
               "what's left for today", "whats left for today", "what remains today",
               "still have for today", "left for today", "remaining today",
               "what else do i have today", "anything left today", "what's remaining","what else", "anything else", "what's left", "whats left",
            "what do i have remaining", "what remains", "what's still there",
            "what do i still got", "what do i got left",]):
            return {"intent": "check_remaining_today", "entities": {},
                    "response_text": f"Let me check what you still have left for today{name_str}."}

        return None

    # ------------------------------------------------------------------
    # LIST QUERIES (tasks, events, reminders)
    # ------------------------------------------------------------------
    def _detect_list_queries(self, msg, msg_lower, user_context):
        name = user_context.get("name", "") if user_context else ""
        name_str = f" {name}" if name else ""

        # Tasks — very broad patterns
        task_patterns = [
            r"^my tasks\b", r"^show my tasks\b", r"^what do i have to do\b",
            r"^my todos\b", r"^list my tasks\b", r"^what are my tasks\b",
            r"^show tasks\b", r"^tasks\b", r"^all my tasks\b",
            r"^what tasks\b", r"^pending tasks\b", r"^my pending tasks\b",
            r"^what do i need to do\b", r"^what should i do\b",
            r"^show me what i have to do\b", r"^what's on my list\b",
            r"^whats on my list\b", r"^my task list\b", r"^task list\b",
            r"^what do i have pending\b", r"^what's pending\b", r"^whats pending\b",
            r"^show me my tasks\b", r"^give me my tasks\b", r"^gimme my tasks\b",
            r"^tell me my tasks\b", r"^what tasks i have\b", r"^what do i got to do\b",
            r"^what i gotta do\b", r"^what i need to do\b", r"^what's my task list\b",
            r"^whats my task list\b", r"^my todo list\b", r"^show todo\b",
            r"^show my todo\b", r"^list tasks\b", r"^task list\b",
            r"^what do i have\b", r"^what i have\b",
            r"^what's there\b", r"^whats there\b",
            r"^what do i got\b", r"^what i got\b",
        ]
        if any(re.search(p, msg_lower) for p in task_patterns):
            return {"intent": "list_tasks", "entities": {},
                    "response_text": f"Here are your tasks{name_str}."}

        # Events — very broad patterns
        event_patterns = [
            r"^my events\b", r"^show my events\b", r"^what's scheduled\b",
            r"^upcoming events\b", r"^check my calendar\b",
            r"^what is in my calendar\b", r"^what's in my calendar\b",
            r"^show my calendar\b", r"^my calendar\b", r"^calendar\b",
            r"^events\b", r"^all my events\b", r"^what events\b",
            r"^upcoming schedule\b", r"^my schedule\b", r"^show schedule\b",
            r"^what's coming up\b", r"^whats coming up\b",
            r"^what am i doing\b", r"^what do i have scheduled\b",
            r"^show me my calendar\b", r"^my upcoming events\b",
            r"^what's on my schedule\b", r"^whats on my schedule\b",
            r"^show me my events\b", r"^give me my events\b", r"^gimme my events\b",
            r"^tell me my events\b", r"^what events i have\b", r"^what's my schedule\b",
            r"^whats my schedule\b", r"^my event list\b", r"^show events\b",
            r"^event list\b", r"^what's on\b", r"^whats on\b",
            r"^what do i have going on\b", r"^what i have going on\b",
            r"^what's happening\b", r"^whats happening\b",
            r"^do i have events\b", r"^any events\b",
            r"^show me what's on\b", r"^show me whats on\b",
        ]
        if any(re.search(p, msg_lower) for p in event_patterns):
            return {"intent": "check_calendar", "entities": {},
                    "response_text": f"Here are your events{name_str}."}

        # Reminders
        rem_patterns = [
            r"^my reminders\b", r"^show my reminders\b",
            r"^list my reminders\b", r"^what reminders\b", r"^reminders\b",
            r"^what am i reminded of\b", r"^my upcoming reminders\b",
            r"^show me my reminders\b", r"^give me my reminders\b",
            r"^tell me my reminders\b", r"^what reminders i have\b",
            r"^do i have reminders\b", r"^any reminders\b",
        ]
        if any(re.search(p, msg_lower) for p in rem_patterns):
            return {"intent": "list_reminders", "entities": {},
                    "response_text": f"Here are your reminders{name_str}."}

        return None

    # ------------------------------------------------------------------
    # SMART PARSERS
    # ------------------------------------------------------------------
    def _try_reminder(self, msg, msg_lower, user_context):
        starters = ["remind me", "remind me to", "don't forget to", "dont forget to",
                    "ping me to", "alert me to", "i need to remember to"]
        is_reminder = any(msg_lower.startswith(s) for s in starters)
        if not is_reminder:
            return None

        title = msg
        for s in starters:
            if title.lower().startswith(s):
                title = title[len(s):].strip()
                break
        title = re.sub(r"^about\s+", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:today|tomorrow|tonight|this evening|next week)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:at|@)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|after|before|by|within)\s+\d+\s*(?:min|minute|hour|hr|day|days|week|weeks|month|months|year|years)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|after|before|by)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:min|minute|hour|hr|day|days|week|weeks|month|months|year|years)\b", "", title, flags=re.IGNORECASE)
        title = title.strip(" ,.!?;:")
        if not title:
            title = "Reminder"
        title = title[0].upper() + title[1:]

        dt_info = self._extract_time(msg_lower, user_context)
        has_time = dt_info[2]
        target = dt_info[3]

        if not has_time and not any(w in msg_lower for w in ["today", "tomorrow", "tonight", "this evening"]):
            return {"intent": "needs_followup",
                    "entities": {"pending_action": "set_reminder", "title": title},
                    "response_text": f"Got it, I'll remind you to {title}. When would you like me to remind you?"}

        if target:
            return {"intent": "set_reminder",
                    "entities": {"title": title, "remind_at": target.strftime("%Y-%m-%dT%H:%M:%S")},
                    "response_text": f"Reminder set: I'll remind you about '{title}' at {target.strftime('%H:%M UTC')}."}

        return {"intent": "needs_followup",
                "entities": {"pending_action": "set_reminder", "title": title},
                "response_text": f"When should I remind you about '{title}'?"}

    def _try_mixed(self, msg, msg_lower, user_context):
        if any(msg_lower.startswith(s) for s in ["remind me", "don't forget", "dont forget", "ping me"]):
            return None

        # FIX: More comprehensive connectors for mixed event+task sentences
        connectors = [
            " and i need to ", " and i have to ", " and i must ", " and i should ",
            " and i want to ", " and i'd like to ", " and i would like to ",
            " and also ", " and ",
            " but ", " then ", " after that ", ", ",
            " plus ", " in addition ", " furthermore ",
        ]
        has_connector = any(c in msg_lower for c in connectors)
        if not has_connector:
            return None

        parts = [msg]
        for c in connectors:
            new_parts = []
            for p in parts:
                # Use case-insensitive split but preserve original case
                import re as _re
                split_parts = _re.split(_re.escape(c), p, flags=_re.IGNORECASE)
                new_parts.extend(split_parts)
            parts = [p.strip() for p in new_parts if len(p.strip()) > 3]

        if len(parts) < 2:
            return None

        actions = []
        for part in parts:
            part_lower = part.lower()
            r = self._try_event(part, part_lower, user_context)
            if not r:
                r = self._try_task(part, part_lower, user_context)
            if not r:
                r = self._try_idea(part, part_lower, user_context)
            if r and r.get("intent") not in ["needs_followup", "ask_followup"]:
                actions.append(r)

        if len(actions) >= 2:
            return {"intent": "multi_action",
                    "entities": {"actions": actions},
                    "response_text": f"Got it! I've added {len(actions)} items for you."}
        return None

    def _try_event(self, msg, msg_lower, user_context):
        event_keywords = ["meeting with", "meeting at", "call with", "dinner with", "lunch with",
                          "appointment with", "appointment at", "schedule for", "schedule ",
                          "go to ", "visit ", "i'm going to ", "i am going to ",
                          "could you schedule", "can you schedule", "please schedule",
                          "schedule a", "schedule an", "meet on", "meeting on",
                          "i have a meeting", "i have an appointment", "i'm meeting",
                          "schedule me"]
        is_event = any(kw in msg_lower for kw in event_keywords)
        if not is_event:
            return None

        title = msg
        for prefix in ["okay so ", "ok so ", "so ", "well ", "yeah ", "yes ", "right ",
                       "could you schedule ", "can you schedule ", "please schedule ",
                       "schedule for ", "schedule me ", "schedule ", "meeting with ", "meeting at ", "call with ",
                       "dinner with ", "lunch with ", "appointment with ", "appointment at ",
                       "go to ", "visit ", "i'm going to ", "i am going to ", "meet on ", "meeting on ",
                       "i have a meeting ", "i have an appointment ", "i'm meeting "]:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
                break

        title = re.sub(r"\b(?:today|tomorrow|tonight|this week|next week)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:at|@)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|after|before|by|on)\s+\d+\s*(?:min|minute|hour|hr|day|days|week|weeks)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|after|before|by|on)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight)\b", "", title, flags=re.IGNORECASE)
        title = title.strip(" ,.!?;:")
        if not title:
            title = "Event"
        title = title[0].upper() + title[1:]

        dt_info = self._extract_time(msg_lower, user_context)
        date, time, has_time, target = dt_info

        people = []
        m = re.search(r"(?:with|and)\s+([A-Za-z]+(?:\s+(?:and|,)\s*[A-Za-z]+)*)", msg_lower)
        if m:
            raw = m.group(1)
            people = [p.strip() for p in re.split(r"\s+(?:and|,)\s*", raw) if len(p.strip()) > 1]

        if not has_time and not any(w in msg_lower for w in ["today", "tomorrow", "tonight", "this evening", "morning", "afternoon", "evening"]):
            return {"intent": "needs_followup",
                    "entities": {"pending_action": "create_event", "title": title, "date": date, "people": people},
                    "response_text": f"I'll schedule '{title}'. What time should I set it for?"}

        return {"intent": "create_event",
                "entities": {"title": title, "date": date, "time": time, "duration": 60, "people": people},
                "response_text": f"Event scheduled: {title} on {date} at {time}."}

    def _try_task(self, msg, msg_lower, user_context):
        # FIX: Check for explanatory/complaint patterns BEFORE treating as task
        explanatory_patterns = [
            r"because you", r"because i", r"because we", r"because they",
            r"you didn't", r"you did not", r"you dont", r"you don't",
            r"i was referring", r"i was talking", r"i'm talking about",
            r"i'm referring to", r"im referring to", r"im talking about",
            r"that's not", r"that is not", r"that isn't",
            r"why did you", r"what happened", r"where is", r"when will",
            r"how come", r"i mean", r"i meant", r"i said",
            r"you messed up", r"you messed", r"you got it wrong",
            r"i'm saying", r"im saying", r"what i mean",
            r"i was asking", r"i asked about", r"i told you",
            r"you forgot", r"you missed", r"you skipped",
            r"did you see", r"did you get", r"did you understand",
            r"i was explaining", r"i was describing", r"i was telling",
        ]
        if any(re.search(p, msg_lower) for p in explanatory_patterns):
            return None

        question_words = ["why", "what", "where", "when", "how", "who", "which"]
        if msg_lower.endswith("?") or any(msg_lower.startswith(w) for w in question_words):
            if not any(s in msg_lower for s in ["need to do", "have to do", "should do", "must do"]):
                return None

        task_starters = ["i need to", "i have to", "i must", "i should", "add task", "new task", "todo:", "to do:", "i want to"]
        is_task = any(msg_lower.startswith(s) for s in task_starters)
                # VOICE FIX: Also check for voice-specific starters that were preprocessed
        voice_starters = ["need to", "have to", "must ", "should ", "want to", "gotta ", "wanna ", "gonna ","i need to", "i have to", "i must", "i should", "i want to", "i gotta", "i wanna", "i am gonna"]
        if not is_task and any(msg_lower.startswith(s) for s in voice_starters):
            is_task = True

        if not is_task and re.search(r"^(i need to do|i have to do|i must do|i should do)\s+my\s+", msg_lower):
            is_task = True

        # "i would like to start reading by tuesday"
        if not is_task and re.search(r"i would like to\s+.+\s+by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)", msg_lower):
            is_task = True
        if not is_task and re.search(r"i'd like to\s+.+\s+by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)", msg_lower):
            is_task = True
        if not is_task and re.search(r"i want to\s+.+\s+by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)", msg_lower):
            is_task = True

        if not is_task:
            action_verbs = ["finish", "complete", "write", "review", "send", "prepare", "buy", "get",
                            "research", "study", "read", "work on", "handle", "fix", "update", "submit",
                            "apply", "call", "email", "organize", "clean", "pay", "workout", "exercise",
                            "cook", "make", "build", "create", "design", "code", "test", "deploy",
                            "start", "begin", "continue", "resume", "plan", "draft", "outline"]
            if any(f" {v} " in f" {msg_lower} " for v in action_verbs):
                if not any(e in msg_lower for e in ["go to ", "visit ", "meeting with", "dinner with", "lunch with", "appointment"]):
                    is_task = True

        if not is_task:
            return None

        # FIX: Extract date context BEFORE splitting, so all tasks get the same date
        date_context = self._extract_time(msg_lower, user_context)
        base_date = date_context[0] if date_context else self.today().strftime("%Y-%m-%d")

        # MULTI-TASK: split on " and " if it looks like multiple tasks
                # MULTI-TASK: split on " and " if it looks like multiple tasks
        if (" and " in msg_lower or ", and " in msg_lower or " also " in msg_lower or ". " in msg_lower) and not any(e in msg_lower for e in ["meeting with", "dinner with", "lunch with", "appointment with"]):
            # Split on multiple connectors: " and ", ", and ", " also ", ". "
            import re as _re
            split_pattern = r'(?:\s+and\s+|\s*,\s*and\s+|\s+also\s+|\.\s+)'
            parts = _re.split(split_pattern, msg)
            parts = [p.strip() for p in parts if len(p.strip()) > 3]
            if len(parts) >= 2:
                actions = []
                for part in parts:
                    part_lower = part.lower()
                    r = self._parse_single_task(part, part_lower, user_context, force_date=base_date , force_starter=True)
                    if r:
                        actions.append(r)
                if len(actions) >= 2:
                    return {"intent": "multi_action",
                            "entities": {"actions": actions},
                            "response_text": f"Got it! I've added {len(actions)} tasks for you."}

        return self._parse_single_task(msg, msg_lower, user_context)
        # ------------------------------------------------------------------
    # VOICE-SPECIFIC TASK DETECTOR — catches imperative/command forms
    # ------------------------------------------------------------------
    def _try_voice_task(self, msg, msg_lower, user_context):
                # Extract date context BEFORE splitting, so all tasks get the same date
        date_context = self._extract_time(msg_lower, user_context)
        base_date = date_context[0] if date_context else self.today().strftime("%Y-%m-%d")
        """Catch voice commands that don't start with 'I need to' etc."""
        voice_task_starters = [
            "work out", "workout", "exercise", "gym",
            "take a shower", "take shower", "shower",
            "finish", "complete", "wrap up", "get done",
            "prepare", "get ready", "make",
            "call", "email", "text", "message",
            "buy", "get", "pick up", "order",
            "read", "study", "review", "go over",
            "send", "submit", "deliver",
            "write", "draft", "compose",
            "clean", "organize", "tidy",
            "pay", "handle", "deal with",
            "start", "begin", "continue",
            "book", "reserve", "schedule",
        ]
        
        starts_with_action = any(msg_lower.startswith(s) for s in voice_task_starters)
        has_and_connector = " and " in msg_lower
        
        if not starts_with_action and not has_and_connector:
            return None
            
        if msg_lower.endswith("?"):
            return None
        if msg_lower.startswith("you ") or msg_lower.startswith("he ") or msg_lower.startswith("she "):
            return None
            
        explanatory_patterns = [
            r"because you", r"because i", r"because we", r"because they",
            r"you didn't", r"you did not", r"you dont", r"you don't",
            r"i was referring", r"i was talking", r"i'm talking about",
            r"i'm referring to", r"im referring to", r"im talking about",
            r"that's not", r"that is not", r"that isn't",
            r"why did you", r"what happened", r"where is", r"when will",
            r"how come", r"i mean", r"i meant", r"i said",
            r"you messed up", r"you messed", r"you got it wrong",
            r"i'm saying", r"im saying", r"what i mean",
            r"i was asking", r"i asked about", r"i told you",
            r"you forgot", r"you missed", r"you skipped",
            r"did you see", r"did you get", r"did you understand",
            r"i was explaining", r"i was describing", r"i was telling",
        ]
        if any(re.search(p, msg_lower) for p in explanatory_patterns):
            return None
            
        event_keywords = ["meeting with", "meeting at", "call with", "dinner with", "lunch with",
                          "appointment with", "appointment at", "go to ", "visit "]
        if any(kw in msg_lower for kw in event_keywords):
            return None
            
        if " and " in msg_lower or ", and " in msg_lower or " also " in msg_lower or ". " in msg_lower:
            import re as _re
            split_pattern = r'(?:\s+and\s+|\s*,\s*and\s+|\s+also\s+|\.\s+)'
            parts = _re.split(split_pattern, msg)
            parts = [p.strip() for p in parts if len(p.strip()) > 3]
            if len(parts) >= 2:
                actions = []
                for part in parts:
                    part_lower = part.lower()
                    r = self._parse_single_task(part, part_lower, user_context, force_starter=True)
                    if r:
                        actions.append(r)
                if len(actions) >= 2:
                    return {"intent": "multi_action",
                            "entities": {"actions": actions},
                            "response_text": f"Got it! I've added {len(actions)} tasks for you."}
                elif len(actions) == 1:
                    return actions[0]
                    
        return self._parse_single_task(msg, msg_lower, user_context, force_starter=True)

    def _parse_single_task(self, msg, msg_lower, user_context, force_date=None, force_starter=False):
        task_starters = ["i need to", "i have to", "i must", "i should", "add task", "new task", "todo:", "to do:", "i want to"]
        title = msg
        for s in task_starters:
            if title.lower().startswith(s):
                title = title[len(s):].strip()
                break
                # VOICE FIX: also strip voice starters
        voice_starters = ["need to ", "have to ", "must ", "should ", "want to ", "gotta ", "wanna ", "gonna ", "imma ", "lemme "]
        for s in voice_starters:
            if title.lower().startswith(s):
                title = title[len(s):].strip()
                break
        if re.search(r"^(i need to do|i have to do|i must do|i should do)\s+my\s+", msg_lower):
            title = re.sub(r"^(i need to do|i have to do|i must do|i should do)\s+my\s+", "", title, flags=re.IGNORECASE)
            title = "Do my " + title.strip()

        # Strip time words, filler words
        title = re.sub(r"\b(?:today|tomorrow|tonight|this week|next week)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:at|by|before|after|on)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|within|after|before|by)\s+\d+\s*(?:min|minute|hour|hr|day|days|week|weeks|month|months|year|years)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b(?:in|after|before|by)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight)\b", "", title, flags=re.IGNORECASE)
        # Strip filler words
        title = re.sub(r"\b(?:too|also|as well|additionally)\b", "", title, flags=re.IGNORECASE)
        title = title.strip(" ,.!?;:")
        if not title:
            title = "Task"
        title = title[0].upper() + title[1:]

        # Smart priority detection
        priority = "medium"
        if any(w in msg_lower for w in ["urgent", "asap", "critical", "important", "deadline", "must", "have to"]):
            priority = "high"
        elif any(w in msg_lower for w in ["maybe", "someday", "whenever", "low priority", "not urgent", "if i have time"]):
            priority = "low"
        # If due is very soon, bump to high
        dt_info = self._extract_time(msg_lower, user_context)
        date = force_date if force_date else dt_info[0]
        if dt_info[3] and dt_info[3].date() <= self.today() + timedelta(days=1):
            if priority == "medium":
                priority = "high"

        return {"intent": "create_task",
                "entities": {"title": title, "priority": priority, "due_date": date},
                "response_text": f"Task added: {title} (Priority: {priority.upper()}, Due: {date})"}

    def _try_idea(self, msg, msg_lower, user_context):
        if msg_lower.startswith("idea:") or msg_lower.startswith("idea "):
            title = msg[5:].strip() if msg_lower.startswith("idea:") else msg[5:].strip()
            title = title[0].upper() + title[1:] if title else "New idea"
            return {"intent": "idea_capture", "entities": {"title": title},
                    "response_text": f"Idea saved: {title}"}

        idea_starters = ["i want to", "i'd like to", "i would like to", "i'm thinking about", "im thinking about",
                         "i bought", "i got", "i plan to", "i've been meaning to", "ive been meaning to"]
        is_idea = any(msg_lower.startswith(s) for s in idea_starters)
        if not is_idea:
            return None

        title = msg
        for s in idea_starters:
            if title.lower().startswith(s):
                title = title[len(s):].strip()
                break
        title = title.strip(" ,.!?;:")
        if not title:
            title = "New idea"
        title = title[0].upper() + title[1:]

        return {"intent": "idea_capture", "entities": {"title": title},
                "response_text": f"I like that direction! I've saved it as an idea: {title}. Want me to turn it into a task?"}

    # ------------------------------------------------------------------
    # TIME & DATE EXTRACTION — REAL DATE AWARE
    # ------------------------------------------------------------------
    def _extract_time(self, msg_lower, user_context):
        now = self.now()
        has_time = False
        target = None
        date_str = self.today().strftime("%Y-%m-%d")
        time_str = "09:00"

        word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                     "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                     "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
                     "thirty": 30, "fifty": 50, "hundred": 100}

        # FIX: Parse afternoon, evening, morning, noon, midnight
        if "afternoon" in msg_lower:
            target = now.replace(hour=15, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (target.strftime("%Y-%m-%d"), "15:00", True, target)
        if "evening" in msg_lower:
            target = now.replace(hour=18, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (target.strftime("%Y-%m-%d"), "18:00", True, target)
        if "morning" in msg_lower:
            target = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (target.strftime("%Y-%m-%d"), "09:00", True, target)
        if "noon" in msg_lower or "midday" in msg_lower:
            target = now.replace(hour=12, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (target.strftime("%Y-%m-%d"), "12:00", True, target)
        if "midnight" in msg_lower:
            target = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (target.strftime("%Y-%m-%d"), "00:00", True, target)

        # Relative numeric: in X minutes/hours/days/weeks/months/years
        m = re.search(r"in\s+(\d+)\s*(min|minute|minutes|hour|hours|hr|hrs|day|days|week|weeks|month|months|year|years)", msg_lower)
        if m:
            amount = int(m.group(1))
            unit = m.group(2)
            has_time = True
            if unit in ["min", "minute", "minutes"]:
                target = now + timedelta(minutes=amount)
            elif unit in ["hour", "hours", "hr", "hrs"]:
                target = now + timedelta(hours=amount)
            elif unit in ["day", "days"]:
                target = now + timedelta(days=amount)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["week", "weeks"]:
                target = now + timedelta(weeks=amount)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["month", "months"]:
                target = now + timedelta(days=amount * 30)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["year", "years"]:
                target = now + timedelta(days=amount * 365)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), target.strftime("%H:%M"), True, target)

        # Relative word: in one week, in two days, in ten days, in one month, in one year
        m = re.search(r"in\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|hundred)\s*(min|minute|minutes|hour|hours|hr|hrs|day|days|week|weeks|month|months|year|years)", msg_lower)
        if m:
            amount = word_nums.get(m.group(1), 1)
            unit = m.group(2)
            has_time = True
            if unit in ["min", "minute", "minutes"]:
                target = now + timedelta(minutes=amount)
            elif unit in ["hour", "hours", "hr", "hrs"]:
                target = now + timedelta(hours=amount)
            elif unit in ["day", "days"]:
                target = now + timedelta(days=amount)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["week", "weeks"]:
                target = now + timedelta(weeks=amount)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["month", "months"]:
                target = now + timedelta(days=amount * 30)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            elif unit in ["year", "years"]:
                target = now + timedelta(days=amount * 365)
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), target.strftime("%H:%M"), True, target)

        # After X days/weeks/months/years
        m = re.search(r"after\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|hundred)\s*(day|days|week|weeks|month|months|year|years)", msg_lower)
        if m:
            amt = word_nums.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else 1)
            unit = m.group(2)
            if "day" in unit:
                target = now + timedelta(days=amt)
            elif "week" in unit:
                target = now + timedelta(weeks=amt)
            elif "month" in unit:
                target = now + timedelta(days=amt * 30)
            else:
                target = now + timedelta(days=amt * 365)
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), target.strftime("%H:%M"), True, target)

        # Before [day]
        m = re.search(r"before\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)", msg_lower)
        if m:
            word = m.group(1)
            if word == "tomorrow":
                target = now + timedelta(days=1)
            elif word == "today":
                target = now
            else:
                target = self._date_from_weekday(word, next_week=False)
                if not target:
                    target = now
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", True, target)

        # By [day]
        m = re.search(r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|this evening)", msg_lower)
        if m:
            word = m.group(1)
            if word == "tomorrow":
                target = now + timedelta(days=1)
            elif word == "today":
                target = now
            elif word in ["tonight", "this evening"]:
                target = now.replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                target = self._date_from_weekday(word, next_week=False)
                if not target:
                    target = now
            if word not in ["tonight", "this evening"]:
                target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), target.strftime("%H:%M"), True, target)

        # On [day]
        m = re.search(r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)", msg_lower)
        if m:
            word = m.group(1)
            if word == "tomorrow":
                target = now + timedelta(days=1)
            elif word == "today":
                target = now
            else:
                target = self._date_from_weekday(word, next_week=False)
                if not target:
                    target = now
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", False, target)

        # Every [day] — recurring (simplified: set for next occurrence)
        m = re.search(r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", msg_lower)
        if m:
            target = self._date_from_weekday(m.group(1), next_week=False)
            if not target:
                target = now
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", False, target)

        # Absolute: at 5pm, at 14:00
        m = re.search(r"(?:at|@)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", msg_lower)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2)) if m.group(2) else 0
            period = m.group(3)
            if period:
                period = period.lower()
                if period == "pm" and hour != 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
            has_time = True
            time_str = f"{hour:02d}:{minute:02d}"
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            date_str = target.strftime("%Y-%m-%d")
            return (date_str, time_str, True, target)

        # Day of week: this Wednesday, next Wednesday, Wednesday
        target = self._parse_weekday(msg_lower)
        if target:
            return (target.strftime("%Y-%m-%d"), "09:00", False, target.replace(hour=9, minute=0, second=0, microsecond=0))

        # Date words
        if "tomorrow" in msg_lower:
            target = now + timedelta(days=1)
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), time_str, has_time, target)

        if "today" in msg_lower:
            target = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if target < now:
                target = now + timedelta(hours=1)
                target = target.replace(minute=0, second=0, microsecond=0)
            return (date_str, time_str, has_time, target)

        if "tonight" in msg_lower or "this evening" in msg_lower:
            target = now.replace(hour=20, minute=0, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            return (date_str, "20:00", True, target)

        if "next week" in msg_lower:
            target = now + timedelta(weeks=1)
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", False, target)

        if "next month" in msg_lower:
            target = now + timedelta(days=30)
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", False, target)

        if "next year" in msg_lower:
            target = now + timedelta(days=365)
            target = target.replace(hour=9, minute=0, second=0, microsecond=0)
            return (target.strftime("%Y-%m-%d"), "09:00", False, target)

        return (date_str, time_str, has_time, target)

    def _parse_weekday(self, msg_lower):
        today = self.today()
        today_num = today.weekday()

        # "next [day]" — next week
        m = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", msg_lower)
        if m:
            day_name = m.group(1)
            day_num = DAY_NUM[day_name]
            days_ahead = day_num - today_num
            if days_ahead <= 0:
                days_ahead += 7
            days_ahead += 7
            result = today + timedelta(days=days_ahead)
            return datetime.combine(result, datetime.min.time()).replace(tzinfo=timezone.utc)

        # "this [day]" — this week (or next if already passed)
        m = re.search(r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", msg_lower)
        if m:
            day_name = m.group(1)
            day_num = DAY_NUM[day_name]
            days_ahead = day_num - today_num
            if days_ahead < 0:
                days_ahead += 7
            result = today + timedelta(days=days_ahead)
            return datetime.combine(result, datetime.min.time()).replace(tzinfo=timezone.utc)

        # Bare [day] — this week if not passed, else next week
        for day_name, day_num in DAY_NUM.items():
            if re.search(rf"\b{day_name}\b", msg_lower):
                days_ahead = day_num - today_num
                if days_ahead <= 0:
                    days_ahead += 7
                result = today + timedelta(days=days_ahead)
                return datetime.combine(result, datetime.min.time()).replace(tzinfo=timezone.utc)

        return None

    def _date_from_weekday(self, day_name, next_week=False):
        today = self.today()
        today_num = today.weekday()
        day_num = DAY_NUM.get(day_name, 0)
        days_ahead = day_num - today_num
        if days_ahead <= 0:
            days_ahead += 7
        if next_week:
            days_ahead += 7
        result = today + timedelta(days=days_ahead)
        return datetime.combine(result, datetime.min.time()).replace(tzinfo=timezone.utc)

    def _parse_date_word(self, word):
        word = word.lower().strip()
        now = self.now()
        if word == "tomorrow":
            return ((now + timedelta(days=1)).strftime("%Y-%m-%d"),)
        if word == "today":
            return (now.strftime("%Y-%m-%d"),)
        if word in DAY_NUM:
            d = self._date_from_weekday(word, next_week=False)
            return (d.strftime("%Y-%m-%d"),) if d else None
        if word == "next week":
            return ((now + timedelta(weeks=1)).strftime("%Y-%m-%d"),)
        if word == "next month":
            return ((now + timedelta(days=30)).strftime("%Y-%m-%d"),)
        if word == "next year":
            return ((now + timedelta(days=365)).strftime("%Y-%m-%d"),)
        return None

    def _parse_relative_date(self, amount_str, unit_str):
        word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                     "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        now = self.now()
        amt = word_nums.get(amount_str, int(amount_str) if amount_str.isdigit() else 1)
        if "day" in unit_str:
            return now + timedelta(days=amt)
        elif "week" in unit_str:
            return now + timedelta(weeks=amt)
        elif "month" in unit_str:
            return now + timedelta(days=amt * 30)
        elif "year" in unit_str:
            return now + timedelta(days=amt * 365)
        return None

    # ------------------------------------------------------------------
    # GEMINI FALLBACK
    # ------------------------------------------------------------------
    def _gemini_process(self, message, user_context, chat_history):
        today_str = self.today().strftime("%Y-%m-%d")
        now_str = self.now().strftime("%H:%M UTC")
        user_info = ""
        if user_context:
            n = user_context.get("name", "the user")
            r = user_context.get("role", "")
            user_info = f"You are Mochi, a personal executive assistant talking to {n}"
            if r:
                user_info += f", a {r}"
            user_info += f". Today is {today_str} and the current time is {now_str}. "

        history_text = ""
        if chat_history:
            recent = chat_history[-6:]
            history_text = "\nRecent chat:\n"
            for h in recent:
                role = "User" if h.get("role") == "user" else "Mochi"
                history_text += f"{role}: {h.get('message', '')}\n"

        system_prompt = (
            user_info +
            "You are Mochi, a warm, conversational personal executive assistant. "
            "You speak naturally, like a helpful friend. You use the user's name when appropriate. "
            "You NEVER say 'I am an AI assistant' or 'I don't have personal experiences'. "
            "You ALWAYS respond as Mochi. You give thoughtful, contextual suggestions. "
            "Return ONLY JSON with keys: intent, entities, response_text. "
            "Intents: create_event, create_task, shared_event, set_reminder, idea_capture, "
            "query_memory, morning_brief_request, list_tasks, check_calendar, check_calendar_date, "
            "check_remaining_today, schedule_briefing, focus_time, stress_check, "
            "mark_all_done, clear_all_tasks, mark_task_by_number, delete_task_by_number, "
            "rename_task, move_task_by_number, change_priority_by_number, "
            "cancel_event, cancel_task, cancel_search, needs_followup, ask_followup, "
            "multi_action, general_chat, wake_word. "
            "CRITICAL: Today is " + today_str + ". Use the real current date. "
            "Always respond as Mochi. Never say 'I am an AI assistant'. "
            "For suggestions, give detailed, contextual advice based on the conversation. "
            + history_text +
            "Return valid JSON only."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json"))
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()
        result = json.loads(text)
        if "intent" not in result:
            return None
        if "entities" not in result:
            result["entities"] = {}
        return result

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def _is_identity_query(self, msg_lower):
        return any(p in msg_lower for p in ["who am i", "do you know who i am", "do you know me",
               "what is my name", "whats my name", "what's my name", "do you remember me"])

    def _identity_response(self, user_context):
        if not user_context:
            return {"intent": "general_chat", "entities": {},
                    "response_text": "I don't know you yet! Send /start to set up your profile."}
        name = user_context.get("name", "there")
        role = user_context.get("role", "")
        tz = user_context.get("timezone", "UTC")
        text = f"You're {name}"
        if role:
            text += f", a {role}"
        text += f". I'm Mochi, your executive assistant. Your timezone is {tz}."
        return {"intent": "general_chat", "entities": {}, "response_text": text}

    def _is_date_query(self, msg_lower):
        return any(p in msg_lower for p in ["what date are we", "what day is it", "what's the date",
               "whats the date", "what date is it", "what day is today", "what's today",
               "whats today", "what is today", "what date is today", "what day is it today",
               "what's the day today", "whats the day today"])

    def _date_response(self, user_context):
        day_name = self.today().strftime("%A")
        date_str = self.today().strftime("%B %d, %Y")
        return {"intent": "general_chat", "entities": {},
                "response_text": f"Today is {day_name}, {date_str}."}

    def _is_wake_word(self, msg_lower):
        return any(re.search(p, msg_lower) for p in [r"^mochi\b", r"^hey mochi\b", r"^ok mochi\b"])

    def _is_morning_brief(self, msg_lower):
        return any(re.search(p, msg_lower) for p in [r"^good morning\b", r"^morning\b",
               r"^what do i have today\b", r"^what's my day\b", r"^show me today\b",
               r"^how's my day\b", r"^how is my day\b", r"^what about my day\b"])

    def _is_schedule_brief(self, msg_lower):
        return any(p in msg_lower for p in ["don't forget to do your morning brief",
               "dont forget your brief", "remember your morning brief", "morning brief tomorrow",
               "send me the brief", "do the brief", "prepare the brief", "set my brief",
               "when is my brief", "briefing time"])

    def _is_suggestion_request(self, msg_lower):
        return any(p in msg_lower for p in ["what do you suggest", "what should i do", "any suggestions",
               "what do you recommend", "help me decide", "what's your advice", "what would you do",
               "what can i do", "any ideas", "what do you think", "do you suggest", "do you recommend",
               "what do you advise", "what's your take", "whats your take", "what do you propose",
               "how should i", "what's the best way", "whats the best way", "how do i approach",
               "what do you think about", "how do you think i should", "can you suggest",
               "could you suggest", "would you suggest", "any recommendation"])

    def _is_focus_time(self, msg_lower):
        return any(p in msg_lower for p in ["protect my time", "focus block", "deep work",
               "dont schedule", "don't schedule", "focus time", "no meetings", "block my calendar",
               "i need to focus", "leave me alone", "do not disturb"])

    def _is_stress_check(self, msg_lower):
        return any(p in msg_lower for p in ["i'm stressed", "im stressed", "overwhelmed",
               "too much", "burned out", "can't handle", "cant handle", "i'm tired", "im tired",
               "exhausted", "drained", "i can't do this", "i cant do this", "too many tasks",
               "too much work", "i need a break"])

    def _resolve_context(self, msg, msg_lower, chat_history):
        if not chat_history or len(chat_history) < 2:
            return None
        if re.search(r"(?:the\s+)?(?:second|2nd|number\s*2|that\s+one)", msg_lower):
            return {"intent": "confirm_selection", "entities": {"selection": 2},
                    "response_text": "Got it, the second one."}
        if re.search(r"^\s*(?:do\s+)?(?:it|that|this)\s*$", msg_lower) or msg_lower in ["do it", "yes do it", "go ahead"]:
            return {"intent": "confirm_last", "entities": {}, "response_text": "I'll proceed with that."}
        return None
