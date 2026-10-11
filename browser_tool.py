"""Bounded live-browser actions for Prototype using Playwright Chromium.

This tool visits public HTTP(S) pages only, blocks private/local destinations,
does not use stored browser profiles or credentials, and returns visible text
rather than exposing arbitrary browser internals.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _public_http_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) webpages are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local network destinations are blocked")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    host, parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            ]
        except OSError as exc:
            raise ValueError("Could not resolve the webpage host") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Private or non-public network destinations are blocked")
    return url


def browser_action(action: str, url: str, target: str = "", timeout_ms: int = 12000) -> dict:
    """Open/read a public page, click visible text, or scroll it, then return a snapshot."""
    action = str(action).strip().lower()
    if action not in {"open", "read", "click", "scroll"}:
        raise ValueError("Action must be open, read, click, or scroll")
    if len(str(url)) > 2000:
        raise ValueError("URL is too long")
    url = _public_http_url(url)
    timeout_ms = max(3000, min(int(timeout_ms), 15000))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Live browser dependency is missing. Run 'python -m pip install -r requirements.txt' "
            "then 'python -m playwright install chromium'."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            java_script_enabled=True,
            accept_downloads=False,
            service_workers="block",
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        def guard_route(route):
            request_url = route.request.url
            parsed = urlparse(request_url)
            if parsed.scheme not in {"http", "https"}:
                route.abort()
                return
            try:
                _public_http_url(request_url)
            except Exception:
                route.abort()
                return
            route.continue_()

        page.route("**/*", guard_route)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if action == "click":
                if not target.strip():
                    raise ValueError("A visible text target is required for click")
                page.get_by_text(target[:200], exact=False).first.click(timeout=timeout_ms)
                page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 5000))
            elif action == "scroll":
                page.evaluate("window.scrollBy(0, Math.max(400, Math.floor(window.innerHeight * 0.8)))")
                page.wait_for_timeout(250)
            elif action == "read":
                page.wait_for_timeout(250)

            snapshot = page.evaluate("""() => {
              const hidden = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','CANVAS','NAV','FOOTER','HEADER','FORM']);
              const root = document.body;
              if (!root) return {text:'', links:[]};
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
              const parts = [];
              let node;
              while ((node = walker.nextNode()) && parts.join(' ').length < 18000) {
                const parent = node.parentElement;
                if (!parent || hidden.has(parent.tagName) || !node.nodeValue.trim()) continue;
                if (parent.closest('script,style,noscript,svg,canvas,nav,footer,header,form,[aria-hidden="true"]')) continue;
                parts.push(node.nodeValue.trim());
              }
              const links = Array.from(root.querySelectorAll('a[href]'))
                .filter(a => a.innerText.trim() && /^https?:$/.test(new URL(a.href, location.href).protocol))
                .slice(0, 25)
                .map(a => ({text:a.innerText.trim().slice(0,160), url:a.href}))
                .filter(a => a.text);
              return {text:parts.join(' ').replace(/\\s+/g,' ').slice(0,18000), links};
            }""")
            return {
                "ok": True,
                "action": action,
                "title": page.title()[:300],
                "url": page.url[:2000],
                "text": snapshot.get("text", ""),
                "links": snapshot.get("links", []),
            }
        finally:
            context.close()
            browser.close()
