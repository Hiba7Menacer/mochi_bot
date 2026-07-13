import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from config.settings import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI
from memory.db import get_user, update_user_calendar_token

# Global storage for auth flows (keyed by user_id to preserve code_verifier)
_auth_flows = {}

class CalendarTool:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.creds = None
        self.service = None
        if user_id:
            self._load_token()

    def _load_token(self):
        user = get_user(self.user_id)
        if user and user.get("calendar_token"):
            try:
                info = json.loads(user["calendar_token"])
                self.creds = Credentials.from_authorized_user_info(info)
                self.service = build("calendar", "v3", credentials=self.creds)
            except Exception as e:
                print(f"[Calendar] Failed to load token: {e}")

    def get_auth_url(self):
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            return None

        # Use InstalledAppFlow for desktop app (shows code on page)
        client_config = {
            "installed": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
            }
        }

        flow = InstalledAppFlow.from_client_config(
            client_config,
            scopes=["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/tasks"],
            redirect_uri="urn:ietf:wg:oauth:2.0:oob"
        )
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

        # CRITICAL: Store the flow object so we can use it later for code exchange
        # This preserves the code_verifier needed for PKCE
        _auth_flows[self.user_id] = flow

        return auth_url

    def exchange_code(self, code):
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise Exception("Google Calendar credentials not configured")

        # CRITICAL: Reuse the SAME flow object that generated the auth URL
        # This preserves the PKCE code_verifier
        flow = _auth_flows.get(self.user_id)

        if flow is None:
            # Fallback: create new flow (may fail with PKCE error, but try anyway)
            client_config = {
                "installed": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
                }
            }
            flow = InstalledAppFlow.from_client_config(
                client_config,
                scopes=["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/tasks"],
                redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )

        flow.fetch_token(code=code)
        self.creds = flow.credentials

        if self.user_id:
            update_user_calendar_token(self.user_id, self.creds.to_json())

        self.service = build("calendar", "v3", credentials=self.creds)

        # Clean up stored flow
        if self.user_id in _auth_flows:
            del _auth_flows[self.user_id]

    def is_authenticated(self):
        return self.service is not None

    def create_task(self, title, due_date=None, notes=""):
        """Create a task in Google Tasks."""
        if not self.creds:
            raise Exception("Not authenticated")
        tasks_service = build("tasks", "v1", credentials=self.creds)
        body = {"title": title}
        if due_date:
            body["due"] = f"{due_date}T00:00:00.000Z"
        if notes:
            body["notes"] = notes
        return tasks_service.tasks().insert(tasklist="@default", body=body).execute()

    def create_event(self, title, date, time, duration=60, attendees=None, timezone_str="UTC"):
        if not self.service:
            raise Exception("Not authenticated")
        start_dt = f"{date}T{time}:00"
        h, m = map(int, time.split(":"))
        end_h = h + duration // 60
        end_m = m + duration % 60
        if end_m >= 60:
            end_h += 1
            end_m -= 60
        end_time = f"{date}T{end_h:02d}:{end_m:02d}:00"
        body = {"summary": title, "start": {"dateTime": start_dt, "timeZone": timezone_str},
                "end": {"dateTime": end_time, "timeZone": timezone_str}}
        if attendees:
            body["attendees"] = [{"email": a} if "@" in a else {"displayName": a} for a in attendees]
        return self.service.events().insert(calendarId="primary", body=body).execute()
