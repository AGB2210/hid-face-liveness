"""Encryption at rest for biometric and secret data (AES-128-CBC + HMAC-SHA256, via Fernet).

Encrypted: face templates, enrollment face crops, recorded test-clip frames, authenticator
(TOTP) secrets and phone numbers. Not encrypted: names, emails, roles and the audit log, which
hold no biometrics (the audit log is protected by its hash chain instead).

Where the key lives, so that a copy of data/ alone is useless:
  HID_DATA_KEY set     the key itself (the AWS server keeps it in /etc/hid-kiosk.env, root-only,
                       outside the data folder)
  Windows              data/keys/data-key.dpapi, sealed with Windows DPAPI: only this Windows
                       account on this machine can unseal it
  otherwise            data/keys/data-key (mode 600), with a warning
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from pipeline.config import DATA

TOKEN_PREFIX = b"gAAAAA"  # every Fernet token starts like this (version byte + timestamp)
SUFFIX = ".enc"
KEY_DIR = DATA / "keys"


# ------------------------------------------------------------------------------ Windows DPAPI
def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def blob(b: bytes) -> tuple[Blob, ctypes.Array]:
        buf = ctypes.create_string_buffer(b, len(b))
        return Blob(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    inp, _keep = blob(data)
    entropy, _keep2 = blob(b"hid-kiosk data key")
    out = Blob()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not fn(ctypes.byref(inp), None, ctypes.byref(entropy), None, None, 0x1, ctypes.byref(out)):  # UI_FORBIDDEN
        raise OSError(f"DPAPI {'seal' if protect else 'unseal'} failed (error {ctypes.GetLastError()}) - "
                      "the data key belongs to another Windows account or machine")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def _load_key() -> bytes:
    env = os.environ.get("HID_DATA_KEY", "").strip()
    if env:
        return env.encode()
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        f = KEY_DIR / "data-key.dpapi"
        if f.exists():
            return _dpapi(f.read_bytes(), protect=False)
        key = Fernet.generate_key()
        f.write_bytes(_dpapi(key, protect=True))
        return key
    f = KEY_DIR / "data-key"
    if not f.exists():
        f.write_bytes(Fernet.generate_key())
        os.chmod(f, 0o600)
    print("[vault] warning: HID_DATA_KEY not set - data key kept in data/keys (same disk as the data)")
    return f.read_bytes().strip()


_f = Fernet(_load_key())


# ------------------------------------------------------------------------------ values
def is_token(b: bytes | str | None) -> bool:
    if b is None:
        return False
    return (b.encode() if isinstance(b, str) else bytes(b)).startswith(TOKEN_PREFIX)


def enc(b: bytes) -> bytes:
    return _f.encrypt(b)


def dec(b: bytes) -> bytes:
    """Decrypt; data written before encryption was added passes through unchanged."""
    return _f.decrypt(bytes(b)) if is_token(b) else bytes(b)


def enc_str(s: str | None) -> str | None:
    return None if s is None else _f.encrypt(s.encode()).decode()


def dec_str(s: str | None) -> str | None:
    return None if s is None else (_f.decrypt(s.encode()).decode() if is_token(s) else s)


def check(sample: bytes | str | None) -> None:
    """Fail at startup, with a clear message, if the key can't open the stored data."""
    if not is_token(sample):
        return
    try:
        _f.decrypt(sample.encode() if isinstance(sample, str) else bytes(sample))
    except InvalidToken:
        raise RuntimeError("The data key does not match the encrypted data in data/ "
                           "(wrong HID_DATA_KEY, or data/ copied from another machine or account)") from None


# ------------------------------------------------------------------------------ sync transport
def _sync_fernet() -> Fernet:
    key = os.environ.get("HID_SYNC_KEY", "")
    if not key:
        raise RuntimeError("HID_SYNC_KEY is not set")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"hid-people-sync|" + key.encode()).digest()))


def seal_for_sync(data: bytes) -> str:
    """Central -> edge payloads (face templates, secrets) are encrypted end to end with a key
    derived from the shared sync key, on top of HTTPS, so no hop in between sees them."""
    return _sync_fernet().encrypt(data).decode()


def open_from_sync(token: str, max_skew_s: int = 900) -> bytes:
    """Stale or replayed payloads are refused (an old list could bring back someone removed).
    15 minutes either way, so a kiosk clock that is a little off still syncs."""
    import time
    f = _sync_fernet()
    data = f.decrypt(token.encode())  # checks integrity first
    age = time.time() - f.extract_timestamp(token.encode())
    if abs(age) > max_skew_s:
        raise ValueError(f"people list is {age:.0f} s old - check this kiosk's clock")
    return data


# ------------------------------------------------------------------------------ files
def write_file(path: Path, data: bytes) -> Path:
    """Write encrypted; the file gets a .enc suffix (frame_00.jpg -> frame_00.jpg.enc)."""
    p = path if path.name.endswith(SUFFIX) else path.with_name(path.name + SUFFIX)
    p.write_bytes(_f.encrypt(data))
    return p


def read_file(path: Path) -> bytes:
    raw = path.read_bytes()
    return _f.decrypt(raw) if path.name.endswith(SUFFIX) else raw


def encrypt_tree(root: Path, pattern: str = "*.jpg") -> int:
    """One-off migration: encrypt plaintext files left from before, then delete the originals."""
    n = 0
    if root.exists():
        for p in root.rglob(pattern):
            write_file(p, p.read_bytes())
            p.unlink()
            n += 1
    return n
