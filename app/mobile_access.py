"""
mobile_access.py — open the app on a phone, with this laptop as the backend.

When the server starts, an ngrok tunnel is opened to the Streamlit port and its public URL is printed
in the terminal together with a QR code; the same QR code is shown in the app ("Open on your phone").

.env settings (all optional):
    NGROK_AUTHTOKEN   free token from https://dashboard.ngrok.com/get-started/your-authtoken; without it
                      the QR code points at the laptop's Wi-Fi address instead (phone on the same network)
    APP_PASSWORD      visitors from outside this laptop must enter it before using the app
    MOBILE_ACCESS=0   no tunnel and no QR code

Streamlit runs app.py only when a browser connects; `streamlit run` opens one on the laptop, so the
QR code appears a few seconds after start (with --server.headless true, after the first visit).
"""

import io
import os
import socket

import streamlit as st


def _lan_ip() -> str | None:
    """This laptop's address on the local network (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def _qr(url: str):
    import qrcode
    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    return qr


def _print_qr(url: str, kind: str) -> None:
    """QR code in the terminal: two modules per character with half blocks, '##' if the console can't show them."""
    rows = _qr(url).get_matrix()
    if len(rows) % 2:
        rows.append([False] * len(rows[0]))
    blocks = {(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}
    # Inverted (light modules drawn) so the code scans on dark terminal backgrounds
    art = "\n".join("".join(blocks[(not a, not b)] for a, b in zip(top, bottom)) for top, bottom in zip(rows[::2], rows[1::2]))
    header = f"\n  Open SEVA AI on your phone ({kind}):\n  {url}\n"
    try:
        print(header + art + "\n", flush=True)
    except UnicodeEncodeError:
        print(header + "\n".join("".join("  " if c else "##" for c in r) for r in rows) + "\n", flush=True)


@st.cache_resource(show_spinner=False)
def phone_url() -> dict | None:
    """Start the tunnel once per server process. Returns {"url", "kind", "note"} or None when disabled."""
    if os.getenv("MOBILE_ACCESS", "1") == "0":
        return None
    port = st.get_option("server.port") or 8501
    token = os.getenv("NGROK_AUTHTOKEN")
    info, note = None, None
    if token:
        try:
            from pyngrok import conf, ngrok
            conf.get_default().auth_token = token
            conf.get_default().monitor_thread = False
            tunnel = ngrok.connect(str(port), "http")  # closed by pyngrok when the server exits
            info = {"url": tunnel.public_url, "kind": "ngrok, works on any network"}
        except Exception as e:
            note = f"ngrok tunnel failed ({str(e).splitlines()[0][:160]}); using the Wi-Fi address instead."
    else:
        note = "NGROK_AUTHTOKEN is not set; the QR code only works on the same Wi-Fi as this laptop."
    if info is None:
        ip = _lan_ip()
        if not ip:
            print(f"[Phone access] {note} No network address found.", flush=True)
            return None
        info = {"url": f"http://{ip}:{port}", "kind": "same Wi-Fi only"}
    info["note"] = note
    if note:
        print(f"[Phone access] {note}", flush=True)
    _print_qr(info["url"], info["kind"])
    return info


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
    """'Open on your phone' panel: QR code and link, for visitors to scan off the laptop screen."""
    info = phone_url()
    if not info:
        return
    with st.expander("Open on your phone", icon=":material/smartphone:", expanded=expanded):
        buf = io.BytesIO()
        _qr(info["url"]).make_image(fill_color="black", back_color="white").save(buf, format="PNG")
        st.image(buf.getvalue(), width=200)
        st.markdown(f"[{info['url']}]({info['url']})")
        st.caption(f"Scan with a phone camera ({info['kind']}).")
