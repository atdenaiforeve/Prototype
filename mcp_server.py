import json
import os
import urllib.error
import urllib.request

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

PROTOTYPE_URL = os.getenv(
    "PROTOTYPE_URL",
    "https://ominous-space-eureka-9qgr546p5v39qp-8000.app.github.dev",
)
PROTOTYPE_TOKEN = os.environ["PROTOTYPE_NOVA_TOKEN"]

NOVA_SESSION_START = "NOVA_SESSION_START_7F3A"
NOVA_SESSION_END = "NOVA_SESSION_END_7F3A"

mcp = MCPServer("Prototype Chat")
session_started = False


def send_to_prototype(message: str) -> dict:
    payload = json.dumps({"message": message}).encode("utf-8")
    request = urllib.request.Request(
        f"{PROTOTYPE_URL}/nova/message",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PROTOTYPE_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Prototype returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Prototype: {exc}") from exc


@mcp.tool()
def prototype_chat(message: str, end_session: bool = False) -> str:
    """Chat with Prototype through the Nova chat bridge."""
    global session_started

    if not session_started:
        send_to_prototype(NOVA_SESSION_START)
        session_started = True

    result = send_to_prototype(message)
    reply = result.get("reply", "")

    if end_session:
        send_to_prototype(NOVA_SESSION_END)
        session_started = False
        return f"{reply}\n[Prototype Nova session ended.]"

    return reply


if __name__ == "__main__":
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8001,
        transport_security=security,
    )
