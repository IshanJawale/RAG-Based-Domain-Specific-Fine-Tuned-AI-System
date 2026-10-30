"""
mobile_access.py — open the app on a phone, with this laptop as the backend.

When the server starts, a public tunnel is opened to the Streamlit port and its link is printed in the
terminal with a QR code; the app shows the same QR codes under "Open on your phone".

Links, in the order they are tried:
  1. Cloudflare quick tunnel (cloudflared): no account and no warning page. Used when cloudflared is
     installed (on PATH, at tools/cloudflared/cloudflared.exe, or CLOUDFLARED_PATH).
  2. ngrok (NGROK_AUTHTOKEN): free plan shows a "You are about to visit" page first. Firewalls such as
     FortiGuard often block *.ngrok-free.app ("Web Filter Violation").
  3. Local network: http://<laptop address>:<port>. Always offered as well. With Windows Mobile hotspot
     on (Settings > Network > Mobile hotspot), a phone that joins the laptop's hotspot reaches it at
     192.168.137.1 without going through any campus firewall, even with no internet at all.

.env settings (all optional):
    TUNNEL            "cloudflare", "ngrok" or "none" to force one (default: the first that works)
    NGROK_AUTHTOKEN   free token from https://dashboard.ngrok.com/get-started/your-authtoken
    APP_PASSWORD      visitors from other devices must enter it before using the app
    MOBILE_ACCESS=0   no tunnel and no QR code

Streamlit runs app.py only when a browser connects; `streamlit run` opens one on the laptop, so the
QR code appears a few seconds after start (with --server.headless true, after the first visit).
"""

import atexit
import io
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
HOTSPOT_IP = "192.168.137.1"  # Windows Mobile hotspot always gives the laptop this address


def _port() -> int:
    return st.get_option("server.port") or 8501


def local_urls() -> list[dict]:
    """Addresses a phone on the same network (or the laptop's hotspot) can use. Re-read on every call,
    so turning the hotspot on after start still shows up in the app."""
    ips = set()
    try:
        ips |= {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # no packets are sent
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    ips = sorted((ip for ip in ips if not ip.startswith(("127.", "169.254."))), key=lambda ip: ip != HOTSPOT_IP)
    return [{"url": f"http://{ip}:{_port()}",
             "label": "Laptop hotspot" if ip == HOTSPOT_IP else f"Same Wi-Fi ({ip})",
             "kind": "phone joined to the laptop's Mobile hotspot" if ip == HOTSPOT_IP else "phone on the same Wi-Fi as this laptop"}
            for ip in ips]


# ----------------------------------------------------------------------
# Tunnels
# ----------------------------------------------------------------------
def _cloudflared() -> str | None:
    # winget's install folders are listed too: a terminal opened before installing has the old PATH
    for p in (os.getenv("CLOUDFLARED_PATH"), ROOT / "tools" / "cloudflared" / "cloudflared.exe",
              ROOT / "tools" / "cloudflared" / "cloudflared", shutil.which("cloudflared"),
              Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "cloudflared" / "cloudflared.exe",
              Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "cloudflared" / "cloudflared.exe"):
        if p and Path(p).is_file():
            return str(p)
    return None


def _start_cloudflare(port: int) -> str:
    exe = _cloudflared()
    if not exe:
        raise RuntimeError("cloudflared is not installed")
    # http2 runs over TCP; the default QUIC (UDP) is often dropped by campus firewalls
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate", "--protocol", "http2"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    atexit.register(proc.terminate)
    found, log = {}, []

    def read():  # keeps draining stderr for the tunnel's lifetime, so the pipe never fills up
        for line in proc.stderr:
            log.append(line.strip())
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and "url" not in found:
                found["url"] = m.group(0)
            if "Registered tunnel connection" in line:
                found["ready"] = True

    threading.Thread(target=read, daemon=True).start()
    deadline = time.time() + 45
    while time.time() < deadline and proc.poll() is None and not ("url" in found and found.get("ready")):
        time.sleep(0.2)
    if "url" in found and found.get("ready"):
        return found["url"]
    proc.terminate()
    last = next((ln for ln in reversed(log) if " ERR " in ln or "error" in ln.lower()), log[-1] if log else "no output")
    raise RuntimeError(f"no tunnel connection within 45 s ({last[-160:]})")


def _start_ngrok(port: int) -> str:
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError("NGROK_AUTHTOKEN is not set")
    from pyngrok import conf, ngrok
    conf.get_default().auth_token = token
    conf.get_default().monitor_thread = False
    return ngrok.connect(str(port), "http").public_url  # closed by pyngrok when the server exits


TUNNELS = {
    "cloudflare": (_start_cloudflare, "Cloudflare tunnel, any network"),
    "ngrok": (_start_ngrok, "ngrok, any network; tap 'Visit Site' on its warning page"),
}


@st.cache_resource(show_spinner=False)
def tunnel() -> dict | None:
    """Open one public tunnel per server process and print the QR codes. Returns {"url", "label", "kind"} or None."""
    if os.getenv("MOBILE_ACCESS", "1") == "0":
        return None
    forced = os.getenv("TUNNEL", "").lower()
    order = [] if forced == "none" else ([forced] if forced in TUNNELS else list(TUNNELS))
    info = None
    for name in order:
        start, kind = TUNNELS[name]
        try:
            info = {"url": start(_port()), "label": f"Internet ({name})", "kind": kind}
            break
        except Exception as e:
            print(f"[Phone access] {name}: {str(e).splitlines()[0][:200]}", flush=True)
    links = ([info] if info else []) + local_urls()
    if links:
        _print_qr(links[0])
        for other in links[1:]:
            print(f"  Also: {other['url']}  ({other['kind']})", flush=True)
    if not info:
        print("[Phone access] No public tunnel: phones must be on the same Wi-Fi or the laptop's hotspot.", flush=True)
    return info


# ----------------------------------------------------------------------
# QR codes
# ----------------------------------------------------------------------
def _qr(url: str):
    import qrcode
    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    return qr


def _print_qr(link: dict) -> None:
    """QR code in the terminal: two modules per character with half blocks, '##' if the console can't show them."""
    rows = _qr(link["url"]).get_matrix()
    if len(rows) % 2:
        rows.append([False] * len(rows[0]))
    blocks = {(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}
    # Inverted (light modules drawn) so the code scans on dark terminal backgrounds
    art = "\n".join("".join(blocks[(not a, not b)] for a, b in zip(top, bottom)) for top, bottom in zip(rows[::2], rows[1::2]))
    header = f"\n  Open SEVA AI on your phone ({link['kind']}):\n  {link['url']}\n"
    try:
        print(header + art + "\n", flush=True)
    except UnicodeEncodeError:
        print(header + "\n".join("".join("  " if c else "##" for c in r) for r in rows) + "\n", flush=True)


# ----------------------------------------------------------------------
# In-app pieces
# ----------------------------------------------------------------------
def _is_local_request() -> bool:
    host = (st.context.headers.get("Host") or "").split(":")[0]
    return host in ("localhost", "127.0.0.1", "[::1]")


def require_password() -> None:
    """Ask visitors (not the laptop itself) for APP_PASSWORD, when one is set."""
    password = os.getenv("APP_PASSWORD")
    if not password or _is_local_request() or st.session_state.get("password_ok"):
        return
    with st.container(horizontal_alignment="center"):
        st.space("large")
        st.subheader(":material/lock: SEVA AI")
        with st.form("password_form", border=True, width=360):
            entered = st.text_input("Password", type="password")
            if st.form_submit_button("Open", type="primary", width="stretch"):
                if entered == password:
                    st.session_state.password_ok = True
                    st.rerun()
                st.error("Wrong password.")
    st.stop()


def render_phone_qr(expanded: bool = False) -> None:
    """'Open on your phone' panel: a QR code per link, for visitors to scan off the laptop screen."""
    if os.getenv("MOBILE_ACCESS", "1") == "0":
        return
    public = tunnel()
    links = ([public] if public else []) + local_urls()
    if not links:
        return
    with st.expander("Open on your phone", icon=":material/smartphone:", expanded=expanded):
        labels = [link["label"] for link in links]
        pick = st.segmented_control("Link", labels, default=labels[0], label_visibility="collapsed",
                                    key="phone_link") if len(links) > 1 else labels[0]
        link = links[labels.index(pick)] if pick in labels else links[0]
        buf = io.BytesIO()
        _qr(link["url"]).make_image(fill_color="black", back_color="white").save(buf, format="PNG")
        st.image(buf.getvalue(), width=200)
        st.markdown(f"[{link['url']}]({link['url']})")
        st.caption(f"Scan with a phone camera ({link['kind']}).")
