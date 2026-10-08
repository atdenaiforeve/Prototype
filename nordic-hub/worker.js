const MAX_MESSAGE_LENGTH = 8000;
const MAX_REQUEST_BYTES = 12000;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=UTF-8",
      "cache-control": "no-store"
    }
  });
}

function unauthorized() {
  return json({ error: "Unauthorized" }, 401);
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type, authorization"
        }
      });
    }

    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/") {
      return json({
        name: "Nordic Hub",
        status: "online",
        protocol: "nordic",
        communication: "ready"
      });
    }

    if (url.pathname !== "/message" || request.method !== "POST") {
      return json({ error: "Not found" }, 404);
    }

    const authorization = request.headers.get("authorization") || "";
    const expected = `Bearer ${env.NORDIC_HUB_SECRET}`;

    if (!env.NORDIC_HUB_SECRET || authorization !== expected) {
      return unauthorized();
    }

    const contentLength = Number(request.headers.get("content-length") || 0);

    if (contentLength > MAX_REQUEST_BYTES) {
      return json({ error: "Request too large" }, 413);
    }

    let body;

    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400);
    }

    const sender = typeof body.sender === "string"
      ? body.sender.trim()
      : "";

    const message = typeof body.message === "string"
      ? body.message.trim()
      : "";

    if (!sender || !message) {
      return json({ error: "sender and message are required" }, 400);
    }

    if (message.length > MAX_MESSAGE_LENGTH) {
      return json({ error: "Message too long" }, 413);
    }

    return json({
      ok: true,
      hub: "Nordic",
      sender,
      message,
      received_at: new Date().toISOString()
    });
  }
};