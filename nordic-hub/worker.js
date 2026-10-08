const MAX_MESSAGE_LENGTH = 8000;
const MAX_REQUEST_BYTES = 12000;
const MAX_NAME_LENGTH = 64;
const MAX_INSTANCE_ID_LENGTH = 128;

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=UTF-8",
      "cache-control": "no-store",
      "access-control-allow-origin": "*"
    }
  });
}

function unauthorized() {
  return json({ error: "Unauthorized" }, 401);
}

function validIdentity(instanceId, name) {
  return (
    typeof instanceId === "string" &&
    typeof name === "string" &&
    instanceId.trim().length > 0 &&
    instanceId.trim().length <= MAX_INSTANCE_ID_LENGTH &&
    name.trim().length > 0 &&
    name.trim().length <= MAX_NAME_LENGTH
  );
}

function registryStub(env) {
  const id = env.NORDIC_REGISTRY.idFromName("global");
  return env.NORDIC_REGISTRY.get(id);
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
        communication: "ready",
        identity_registry: "online"
      });
    }

    const authorization = request.headers.get("authorization") || "";
    const expected = `Bearer ${env.NORDIC_HUB_SECRET}`;

    if (!env.NORDIC_HUB_SECRET || authorization !== expected) {
      return unauthorized();
    }

    if (url.pathname === "/message" && request.method === "POST") {
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

      const sender = typeof body.sender === "string" ? body.sender.trim() : "";
      const message = typeof body.message === "string" ? body.message.trim() : "";

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

    if (url.pathname === "/register" && request.method === "POST") {
      let body;

      try {
        body = await request.json();
      } catch {
        return json({ error: "Invalid JSON" }, 400);
      }

      const instanceId = typeof body.instance_id === "string" ? body.instance_id.trim() : "";
      const name = typeof body.name === "string" ? body.name.trim() : "";

      if (!validIdentity(instanceId, name)) {
        return json({
          error: "instance_id and name are required and within allowed lengths"
        }, 400);
      }

      return registryStub(env).fetch("https://registry/register", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ instance_id: instanceId, name })
      });
    }

    if (url.pathname === "/heartbeat" && request.method === "POST") {
      let body;

      try {
        body = await request.json();
      } catch {
        return json({ error: "Invalid JSON" }, 400);
      }

      const instanceId = typeof body.instance_id === "string" ? body.instance_id.trim() : "";

      if (!instanceId || instanceId.length > MAX_INSTANCE_ID_LENGTH) {
        return json({ error: "Valid instance_id is required" }, 400);
      }

      return registryStub(env).fetch("https://registry/heartbeat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ instance_id: instanceId })
      });
    }

    if (url.pathname === "/identity" && request.method === "GET") {
      const instanceId = (url.searchParams.get("instance_id") || "").trim();

      if (!instanceId || instanceId.length > MAX_INSTANCE_ID_LENGTH) {
        return json({ error: "Valid instance_id is required" }, 400);
      }

      return registryStub(env).fetch(
        `https://registry/identity?instance_id=${encodeURIComponent(instanceId)}`
      );
    }

    return json({ error: "Not found" }, 404);
  }
};

export class NordicRegistry {
  constructor(state) {
    this.state = state;
  }

  async fetch(request) {
    const url = new URL(request.url);
    const identities = (await this.state.storage.get("identities")) || {};

    if (request.method === "POST" && url.pathname === "/register") {
      const body = await request.json();
      const instanceId = body.instance_id;
      const name = body.name;

      const existing = identities[instanceId];
      const now = new Date().toISOString();

      if (existing) {
        existing.name = name;
        existing.last_seen = now;
        existing.known = true;
        await this.state.storage.put("identities", identities);

        return json({
          ok: true,
          registered: false,
          known: true,
          identity: existing
        });
      }

      const identity = {
        instance_id: instanceId,
        name,
        first_seen: now,
        last_seen: now,
        known: true
      };

      identities[instanceId] = identity;
      await this.state.storage.put("identities", identities);

      return json({
        ok: true,
        registered: true,
        known: true,
        identity
      });
    }

    if (request.method === "POST" && url.pathname === "/heartbeat") {
      const instanceId = bodyValue(await request.json(), "instance_id");
      const identity = identities[instanceId];

      if (!identity) {
        return json({ error: "Unknown instance_id" }, 404);
      }

      identity.last_seen = new Date().toISOString();
      await this.state.storage.put("identities", identities);

      return json({ ok: true, identity });
    }

    if (request.method === "GET" && url.pathname === "/identity") {
      const instanceId = (url.searchParams.get("instance_id") || "").trim();
      const identity = identities[instanceId];

      if (!identity) {
        return json({ error: "Unknown instance_id" }, 404);
      }

      return json({ ok: true, identity });
    }

    return json({ error: "Not found" }, 404);
  }
}

function bodyValue(body, key) {
  return typeof body?.[key] === "string" ? body[key].trim() : "";
}
