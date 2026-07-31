from __future__ import annotations
import json
import re
import string
import time
import uuid
import random
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
SUPABASE_URL = "https://auth.gratisfy.xyz"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNjQxNzY5MjAwLCJleHAiOjQxMDI0NDQ4MDB9.GdZl3P5sG09Vlq15A4TIAIWHUyCdFf8rdOx4B_zYuCQ"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
DEFAULT_OUTPUT = Path(__file__).parent / "credentials.json"
REFRESH_LEEWAY_S = 60.0
HTTP_TIMEOUT = 30
MAILTM_BASE = "https://api.mail.tm"
MAILTM_TIMEOUT = 30
MAILTM_VERIFY_POLL_SECS = 5.0
MAILTM_VERIFY_TIMEOUT_SECS = 180.0
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
@dataclass
class Credential:
    id: str
    email: Optional[str] = None
    password: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: float = 0.0
    user_agent: str = DEFAULT_USER_AGENT
    created_at: str = field(default_factory=_utc_now)
    last_refreshed_at: Optional[str] = None
    source: str = "user"
    def is_valid(self) -> bool:
        return bool(self.access_token and self.refresh_token)
    def needs_refresh(self) -> bool:
        if not self.access_token:
            return True
        return time.time() >= (self.expires_at - REFRESH_LEEWAY_S)
    def bearer_header(self) -> str:
        if not self.access_token:
            raise RuntimeError("credential is missing access_token; call refresh()")
        return f"Bearer {self.access_token}"
    def update_token(self, *, access_token: str, refresh_token: Optional[str] = None, expires_in: Optional[int] = None) -> None:
        self.access_token = access_token
        if refresh_token:
            self.refresh_token = refresh_token
        if expires_in is not None:
            self.expires_at = time.time() + max(30, int(expires_in) - 10)
        self.last_refreshed_at = _utc_now()
class CredentialFile:
    def __init__(self, path: Path = DEFAULT_OUTPUT) -> None:
        self.path = Path(path)
    def load(self) -> List[Credential]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        items = raw.get("credentials", []) if isinstance(raw, dict) else raw
        creds: List[Credential] = []
        for item in items:
            creds.append(Credential(
                id=item.get("id") or str(uuid.uuid4()),
                email=item.get("email"),
                password=item.get("password"),
                access_token=item.get("access_token"),
                refresh_token=item.get("refresh_token"),
                expires_at=float(item.get("expires_at", 0.0) or 0.0),
                user_agent=item.get("user_agent", DEFAULT_USER_AGENT),
                created_at=item.get("created_at", _utc_now()),
                last_refreshed_at=item.get("last_refreshed_at"),
                source=item.get("source", "user"),
            ))
        return creds
    def save(self, credentials: List[Credential]) -> None:
        payload = {
            "version": 1,
            "updated_at": _utc_now(),
            "count": len(credentials),
            "credentials": [asdict(c) for c in credentials],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
class _HttpError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body
def _auth_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}{path}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    with requests.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT) as resp:
        text = resp.text
        if not resp.ok:
            raise _HttpError(f"auth {resp.status_code}: {text[:300]}", status=resp.status_code, body=text)
        try:
            return json.loads(text) if text else {}
        except Exception as exc:
            raise _HttpError(f"auth response not json: {exc}", status=resp.status_code, body=text)
def refresh_via_refresh_token(credential: Credential) -> Credential:
    if not credential.refresh_token:
        raise RuntimeError("credential has no refresh_token")
    body = {"refresh_token": credential.refresh_token}
    data = _auth_post("/auth/v1/token?grant_type=refresh_token", body)
    access = data.get("access_token")
    if not access:
        raise RuntimeError("refresh response missing access_token")
    credential.update_token(
        access_token=access,
        refresh_token=data.get("refresh_token") or credential.refresh_token,
        expires_in=data.get("expires_in"),
    )
    return credential
def refresh_via_password(credential: Credential) -> Credential:
    if not (credential.email and credential.password):
        raise RuntimeError("credential needs password for password grant")
    body = {"email": credential.email, "password": credential.password}
    data = _auth_post("/auth/v1/token?grant_type=password", body)
    access = data.get("access_token")
    if not access:
        raise RuntimeError("password response missing access_token")
    credential.update_token(
        access_token=access,
        refresh_token=data.get("refresh_token") or credential.refresh_token,
        expires_in=data.get("expires_in"),
    )
    return credential
def refresh(credential: Credential) -> Credential:
    if credential.refresh_token:
        try:
            return refresh_via_refresh_token(credential)
        except _HttpError as exc:
            if exc.status in (400, 401) and credential.email and credential.password:
                return refresh_via_password(credential)
            raise
    if credential.email and credential.password:
        return refresh_via_password(credential)
    raise RuntimeError("credential cannot be refreshed (no refresh_token or email/password)")
def ensure_fresh(credential: Credential) -> Credential:
    if not credential.needs_refresh():
        return credential
    return refresh(credential)
def _mailtm_headers(token: Optional[str] = None) -> Dict[str, str]:
    h = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/ld+json, application/json",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h
def _mailtm_get_active_domain() -> str:
    with requests.get(f"{MAILTM_BASE}/domains", headers=_mailtm_headers(), timeout=MAILTM_TIMEOUT) as r:
        r.raise_for_status()
        data = r.json()
    members = data.get("hydra:member") or data.get("hydra:members") or []
    for m in members:
        if m.get("isActive"):
            return m["domain"]
    if members:
        return members[0]["domain"]
    raise RuntimeError("no active mail.tm domain")
def _mailtm_create_account(address: str, password: str) -> Dict[str, Any]:
    body = {"address": address, "password": password}
    with requests.post(f"{MAILTM_BASE}/accounts", json=body, headers=_mailtm_headers(), timeout=MAILTM_TIMEOUT) as r:
        if r.status_code == 409:
            raise RuntimeError(f"mail.tm account collision for {address}")
        r.raise_for_status()
        return r.json()
def _mailtm_get_token(address: str, password: str) -> str:
    body = {"address": address, "password": password}
    with requests.post(f"{MAILTM_BASE}/token", json=body, headers=_mailtm_headers(), timeout=MAILTM_TIMEOUT) as r:
        r.raise_for_status()
        return r.json()["token"]
def _mailtm_list_messages(token: str) -> List[Dict[str, Any]]:
    with requests.get(f"{MAILTM_BASE}/messages", headers=_mailtm_headers(token), timeout=MAILTM_TIMEOUT) as r:
        r.raise_for_status()
        data = r.json()
    return data.get("hydra:member") or []
def _mailtm_read_message(token: str, message_id: str) -> Dict[str, Any]:
    with requests.get(f"{MAILTM_BASE}/messages/{message_id}", headers=_mailtm_headers(token), timeout=MAILTM_TIMEOUT) as r:
        r.raise_for_status()
        return r.json()
def _extract_verify_url(msg: Dict[str, Any]) -> Optional[str]:
    text_parts = [msg.get("text") or "", msg.get("intro") or ""]
    html = msg.get("html") or ""
    if isinstance(html, list):
        html = " ".join(str(p) for p in html)
    text_parts.append(html)
    body = "\n".join(text_parts)
    m = re.search(r'https://auth\.gratisfy\.xyz/auth/v1/verify\?[^"\'<>\s]+', body)
    if m:
        url = m.group(0)
        url = url.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
        return url
    return None
def _mailtm_wait_for_verify_url(token: str, timeout_s: float = MAILTM_VERIFY_TIMEOUT_SECS) -> str:
    deadline = time.time() + timeout_s
    seen_ids: set[str] = set()
    while time.time() < deadline:
        for msg in _mailtm_list_messages(token):
            mid = msg.get("id")
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
            full = _mailtm_read_message(token, mid)
            url = _extract_verify_url(full)
            if url:
                return url
        time.sleep(MAILTM_VERIFY_POLL_SECS)
    raise RuntimeError(f"no verify email received within {timeout_s}s")
def _gratisfy_signup(email: str, password: str) -> Dict[str, Any]:
    body = {"email": email, "password": password}
    with requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        json=body,
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        timeout=HTTP_TIMEOUT,
    ) as r:
        if not r.ok:
            raise RuntimeError(f"signup failed {r.status_code}: {r.text[:300]}")
        return r.json()
def _gratisfy_password_login(email: str, password: str) -> Dict[str, Any]:
    body = {"email": email, "password": password}
    with requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        json=body,
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        timeout=HTTP_TIMEOUT,
    ) as r:
        if not r.ok:
            raise RuntimeError(f"login failed {r.status_code}: {r.text[:300]}")
        return r.json()
def _confirm_email(verify_url: str) -> None:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    with requests.get(verify_url, headers=headers, allow_redirects=True, timeout=HTTP_TIMEOUT) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"verify failed {r.status_code}: {r.text[:200]}")
def _rand_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    return "".join(random.choice(alphabet) for _ in range(length))
def _rand_username(prefix: str = "pa") -> str:
    suffix = uuid.uuid4().hex[:10]
    return f"{prefix}_{suffix}"
def harvest_one(mailtm_domain: Optional[str] = None) -> Credential:
    domain = (mailtm_domain or "").strip() or _mailtm_get_active_domain()
    mail_user = _rand_username()
    mail_pw = _rand_password(20)
    grat_user = _rand_username()
    grat_pw = _rand_password()
    email = f"{mail_user}@{domain}"
    _mailtm_create_account(email, mail_pw)
    mail_token = _mailtm_get_token(email, mail_pw)
    _gratisfy_signup(email, grat_pw)
    verify_url = _mailtm_wait_for_verify_url(mail_token)
    _confirm_email(verify_url)
    login = _gratisfy_password_login(email, grat_pw)
    cred = Credential(
        id=str(uuid.uuid4()),
        email=email,
        password=grat_pw,
        access_token=login["access_token"],
        refresh_token=login.get("refresh_token"),
        expires_at=time.time() + max(30, int(login.get("expires_in", 3600)) - 30),
        source="auto-harvest",
    )
    return cred
def harvest(count: int, output: Path = DEFAULT_OUTPUT, on_event=None) -> List[Credential]:
    store = CredentialFile(output)
    existing = store.load()
    harvested: List[Credential] = []
    for i in range(count):
        try:
            cred = harvest_one()
        except Exception as exc:
            if on_event:
                on_event("error", i + 1, count, str(exc))
            continue
        if not cred.is_valid():
            if on_event:
                on_event("skip", i + 1, count, None)
            continue
        harvested.append(cred)
        store.save(existing + harvested)
        if on_event:
            on_event("ok", i + 1, count, cred.email)
    return harvested
def import_from_supabase_localstorage(text: str) -> Credential:
    if isinstance(text, dict):
        data = text
    else:
        data = json.loads(text)
    access = data.get("access_token")
    refresh_tok = data.get("refresh_token")
    if not access or not refresh_tok:
        raise ValueError("payload missing access_token or refresh_token")
    expires_at = float(data.get("expires_at") or 0.0)
    cred = Credential(
        id=str(uuid.uuid4()),
        email=data.get("user", {}).get("email"),
        access_token=access,
        refresh_token=refresh_tok,
        expires_at=expires_at,
        source="browser-paste",
    )
    return cred
def add_credentials(count_or_payloads, output: Path = DEFAULT_OUTPUT) -> List[Credential]:
    store = CredentialFile(output)
    existing = store.load()
    if isinstance(count_or_payloads, int):
        return existing
    added: List[Credential] = []
    for item in count_or_payloads or []:
        try:
            cred = import_from_supabase_localstorage(item)
        except Exception:
            if isinstance(item, dict) and item.get("email") and item.get("password"):
                cred = Credential(id=str(uuid.uuid4()), email=item["email"], password=item["password"], source="email-pass")
            else:
                raise
        added.append(cred)
    store.save(existing + added)
    return existing + added
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Gratisfy credential pool manager")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list credentials")
    sub.add_parser("refresh", help="force-refresh any expired credentials")
    p_harvest = sub.add_parser("harvest", help="provision new credentials via temp-mail")
    p_harvest.add_argument("-n", "--count", type=int, default=1)
    p_harvest.add_argument("--domain", default=None, help="mail.tm domain override (default: active domain)")
    args = ap.parse_args()
    cmd = args.cmd or "list"
    store = CredentialFile()
    if cmd == "list":
        creds = store.load()
        print(f"loaded {len(creds)} credential(s)")
        for i, c in enumerate(creds, 1):
            print(
                f"  {i}/{len(creds)} id={c.id} email={c.email or '<unknown>'} "
                f"expires={datetime.fromtimestamp(c.expires_at).isoformat() if c.expires_at else 'never'} "
                f"needs_refresh={c.needs_refresh()}"
            )
        return 0
    if cmd == "refresh":
        creds = store.load()
        updated = 0
        for c in creds:
            if c.needs_refresh():
                try:
                    refresh(c)
                    updated += 1
                except Exception as exc:
                    print(f"  refresh failed for {c.email or c.id}: {exc}")
        store.save(creds)
        print(f"refreshed {updated}/{len(creds)} credential(s)")
        return 0
    if cmd == "harvest":
        def _ev(kind, i, n, detail):
            if kind == "ok":
                print(f"{i}/{n} ok {detail}")
            elif kind == "skip":
                print(f"{i}/{n} skip")
            else:
                print(f"{i}/{n} fail {detail}")
        harvested = harvest(args.count, on_event=_ev)
        print(f"harvested {len(harvested)} credential(s)")
        return 0
    return 1
if __name__ == "__main__":
    raise SystemExit(main())