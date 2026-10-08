```javascript
const MAX_MESSAGE_LENGTH = 8000;
const MAX_REQUEST_BYTES = 12000;
const MAX_NAME_LENGTH = 64;
const MAX_INSTANCE_ID_LENGTH = 128;
const MAX_TOKEN_LENGTH = 128;

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

function validToken(token) {
  return (
    typeof token === "string" &&
    token.length > 0 &&
    token.length <= MAX_TOKEN_LENGTH
  );
}

function registryStub(env) {
  const id = env.NORDIC_REGISTRY.idFromName("global");
  return env.NORDIC_REGISTRY.get(id);
}

function bodyValue(body, key) {
  return typeof body?.[key] === "string" ? body[key].trim() : "";
}

async function readJson(request) {
  const contentLength = Number(
    request.headers.get("content-length") || 0
  );

  if (contentLength > MAX_REQUEST_BYTES) {
    throw new Error("Request too large");
  }

  return request.json();
}

async function queueForInstance(
  env,
  instanceId,
  sender,
  message
) {
  return registryStub(env).fetch("https://registry/queue", {
    method: "POST",
    headers: {
      "content-type": "application/json"
    },
    body: JSON.stringify({
      instance_id: instanceId,
      sender,
      message
    })
  });
}

export default {
  async fetch(request, env) {
    /*
     * CORS preflight
     */
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers":
            "content-type, authorization"
        }
      });
    }

    const url = new URL(request.url);

    /*
     * Public health/status endpoint
     */
    if (request.method === "GET" && url.pathname === "/") {
      return json({
        name: "Nordic Hub",
        status: "online",
        protocol: "nordic",
        communication: "ready",
        identity_registry: "online",
        instance_inboxes: "online",
        mcp: "not-publicly-writable"
      });
    }

    /*
     * Instance registration is public.
     *
     * Each Prototype instance receives its own private instance token.
     * The master Nordic Hub secret is never sent to the browser.
     */
    if (
      url.pathname === "/register" &&
      request.method === "POST"
    ) {
      let body;

      try {
        body = await readJson(request);
      } catch (error) {
        return json(
          {
            error:
              error.message === "Request too large"
                ? error.message
                : "Invalid JSON"
          },
          error.message === "Request too large" ? 413 : 400
        );
      }

      const instanceId = bodyValue(body, "instance_id");
      const name = bodyValue(body, "name");

      if (!validIdentity(instanceId, name)) {
        return json(
          {
            error:
              "instance_id and name are required and within allowed lengths"
          },
          400
        );
      }

      return registryStub(env).fetch(
        "https://registry/register",
        {
          method: "POST",
          headers: {
            "content-type": "application/json"
          },
          body: JSON.stringify({
            instance_id: instanceId,
            name
          })
        }
      );
    }

    /*
     * The following server-side endpoints require the master
     * Nordic Hub secret.
     */
    const authorization =
      request.headers.get("authorization") || "";

    const expected =
      env.NORDIC_HUB_SECRET || "";

    if (
      !expected ||
      authorization !== `Bearer ${expected}`
    ) {
      /*
       * Browser Prototype instances use their own instance
       * token for /pull and /instance-heartbeat.
       */
      if (
        url.pathname !== "/pull" &&
        url.pathname !== "/instance-heartbeat"
      ) {
        return unauthorized();
      }
    }

    /*
     * Send a message through Nordic.
     *
     * If instance_id is supplied, the message is placed into
     * that Prototype instance's inbox.
     */
    if (
      url.pathname === "/message" &&
      request.method === "POST"
    ) {
      let body;

      try {
        body = await readJson(request);
      } catch (error) {
        return json(
          {
            error:
              error.message === "Request too large"
                ? error.message
                : "Invalid JSON"
          },
          error.message === "Request too large" ? 413 : 400
        );
      }

      const sender = bodyValue(body, "sender");
      const message = bodyValue(body, "message");
      const instanceId = bodyValue(body, "instance_id");

      if (!sender || !message) {
        return json(
          {
            error:
              "sender and message are required"
          },
          400
        );
      }

      if (message.length > MAX_MESSAGE_LENGTH) {
        return json(
          {
            error: "Message too long"
          },
          413
        );
      }

      if (instanceId) {
        return queueForInstance(
          env,
          instanceId,
          sender,
          message
        );
      }

      return json({
        ok: true,
        hub: "Nordic",
        sender,
        message,
        received_at:
          new Date().toISOString()
      });
    }

    /*
     * Server-side heartbeat.
     */
    if (
      url.pathname === "/heartbeat" &&
      request.method === "POST"
    ) {
      let body;

      try {
        body = await request.json();
      } catch {
        return json(
          {
            error: "Invalid JSON"
          },
          400
        );
      }

      const instanceId =
        bodyValue(body, "instance_id");

      if (
        !instanceId ||
        instanceId.length >
          MAX_INSTANCE_ID_LENGTH
      ) {
        return json(
          {
            error:
              "Valid instance_id is required"
          },
          400
        );
      }

      return registryStub(env).fetch(
        "https://registry/heartbeat",
        {
          method: "POST",
          headers: {
            "content-type": "application/json"
          },
          body: JSON.stringify({
            instance_id: instanceId
          })
        }
      );
    }

    /*
     * Look up an identity.
     */
    if (
      url.pathname === "/identity" &&
      request.method === "GET"
    ) {
      const instanceId =
        (
          url.searchParams.get(
            "instance_id"
          ) || ""
        ).trim();

      if (
        !instanceId ||
        instanceId.length >
          MAX_INSTANCE_ID_LENGTH
      ) {
        return json(
          {
            error:
              "Valid instance_id is required"
          },
          400
        );
      }

      return registryStub(env).fetch(
        `https://registry/identity?instance_id=${encodeURIComponent(
          instanceId
        )}`
      );
    }

    /*
     * Prototype browser pulls its messages from its
     * own private inbox.
     */
    if (
      url.pathname === "/pull" &&
      request.method === "POST"
    ) {
      let body;

      try {
        body = await readJson(request);
      } catch {
        return json(
          {
            error: "Invalid JSON"
          },
          400
        );
      }

      const instanceId =
        bodyValue(body, "instance_id");

      const token =
        bodyValue(body, "token");

      if (
        !instanceId ||
        !validToken(token)
      ) {
        return json(
          {
            error:
              "instance_id and token are required"
          },
          400
        );
      }

      return registryStub(env).fetch(
        "https://registry/pull",
        {
          method: "POST",
          headers: {
            "content-type": "application/json"
          },
          body: JSON.stringify({
            instance_id: instanceId,
            token
          })
        }
      );
    }

    /*
     * Browser Prototype heartbeat using its own
     * private instance token.
     */
    if (
      url.pathname === "/instance-heartbeat" &&
      request.method === "POST"
    ) {
      let body;

      try {
        body = await readJson(request);
      } catch {
        return json(
          {
            error: "Invalid JSON"
          },
          400
        );
      }

      const instanceId =
        bodyValue(body, "instance_id");

      const token =
        bodyValue(body, "token");

      if (
        !instanceId ||
        !validToken(token)
      ) {
        return json(
          {
            error:
              "instance_id and token are required"
          },
          400
        );
      }

      return registryStub(env).fetch(
        "https://registry/instance-heartbeat",
        {
          method: "POST",
          headers: {
            "content-type": "application/json"
          },
          body: JSON.stringify({
            instance_id: instanceId,
            token
          })
        }
      );
    }

    return json(
      {
        error: "Not found"
      },
      404
    );
  }
};


/*
 * Persistent Nordic identity registry.
 *
 * Every Prototype instance has:
 *
 *   instance_id  = unique persistent identity
 *   name         = chosen AI name
 *   instance_token = private token for that instance
 *
 * The name and ID remain separate.
 */
export class NordicRegistry {
  constructor(state) {
    this.state = state;
  }

  async fetch(request) {
    const url = new URL(request.url);

    const identities =
      (await this.state.storage.get(
        "identities"
      )) || {};

    const queues =
      (await this.state.storage.get(
        "queues"
      )) || {};

    /*
     * Register or reconnect an instance.
     */
    if (
      request.method === "POST" &&
      url.pathname === "/register"
    ) {
      const body =
        await request.json();

      const instanceId =
        body.instance_id;

      const name =
        body.name;

      const existing =
        identities[instanceId];

      const now =
        new Date().toISOString();

      /*
       * Existing instance:
       * preserve its identity and token.
       */
      if (existing) {
        existing.name = name;
        existing.last_seen = now;
        existing.known = true;

        await this.state.storage.put(
          "identities",
          identities
        );

        if (!queues[instanceId]) {
          queues[instanceId] = [];

          await this.state.storage.put(
            "queues",
            queues
          );
        }

        return json({
          ok: true,
          registered: false,
          known: true,
          identity: publicIdentity(
            existing,
            true
          )
        });
      }

      /*
       * New instance gets a unique private token.
       */
      const instanceToken =
        crypto.randomUUID();

      const identity = {
        instance_id: instanceId,
        name,
        first_seen: now,
        last_seen: now,
        known: true,
        instance_token:
          instanceToken
      };

      identities[instanceId] =
        identity;

      queues[instanceId] = [];

      await this.state.storage.put(
        "identities",
        identities
      );

      await this.state.storage.put(
        "queues",
        queues
      );

      return json({
        ok: true,
        registered: true,
        known: true,
        identity
      });
    }

    /*
     * Server-side heartbeat.
     */
    if (
      request.method === "POST" &&
      url.pathname === "/heartbeat"
    ) {
      const instanceId =
        bodyValue(
          await request.json(),
          "instance_id"
        );

      const identity =
        identities[instanceId];

      if (!identity) {
        return json(
          {
            error:
              "Unknown instance_id"
          },
          404
        );
      }

      identity.last_seen =
        new Date().toISOString();

      await this.state.storage.put(
        "identities",
        identities
      );

      return json({
        ok: true,
        identity:
          publicIdentity(identity)
      });
    }

    /*
     * Browser instance heartbeat.
     */
    if (
      request.method === "POST" &&
      url.pathname ===
        "/instance-heartbeat"
    ) {
      const body =
        await request.json();

      const instanceId =
        bodyValue(
          body,
          "instance_id"
        );

      const token =
        bodyValue(
          body,
          "token"
        );

      const identity =
        identities[instanceId];

      if (!identity) {
        return json(
          {
            error:
              "Unknown instance_id"
          },
          404
        );
      }

      if (
        identity.instance_token !==
        token
      ) {
        return unauthorized();
      }

      identity.last_seen =
        new Date().toISOString();

      await this.state.storage.put(
        "identities",
        identities
      );

      return json({
        ok: true,
        identity:
          publicIdentity(identity)
      });
    }

    /*
     * Add a message to a Prototype inbox.
     */
    if (
      request.method === "POST" &&
      url.pathname === "/queue"
    ) {
      const body =
        await request.json();

      const instanceId =
        bodyValue(
          body,
          "instance_id"
        );

      const sender =
        bodyValue(
          body,
          "sender"
        );

      const message =
        bodyValue(
          body,
          "message"
        );

      if (
        !identities[instanceId]
      ) {
        return json(
          {
            error:
              "Unknown instance_id"
          },
          404
        );
      }

      if (
        !sender ||
        !message
      ) {
        return json(
          {
            error:
              "sender and message are required"
          },
          400
        );
      }

      if (
        message.length >
        MAX_MESSAGE_LENGTH
      ) {
        return json(
          {
            error:
              "Message too long"
          },
          413
        );
      }

      const queue =
        queues[instanceId] || [];

      queue.push({
        id:
          crypto.randomUUID(),
        sender,
        message,
        created_at:
          new Date().toISOString()
      });

      /*
       * Keep only the latest 100 messages.
       */
      queues[instanceId] =
        queue.slice(-100);

      await this.state.storage.put(
        "queues",
        queues
      );

      return json({
        ok: true,
        queued: true,
        instance_id:
          instanceId,
        queue_size:
          queues[instanceId].length
      });
    }

    /*
     * Prototype retrieves its inbox.
     *
     * Messages are removed after being pulled so they
     * aren't repeatedly delivered.
     */
    if (
      request.method === "POST" &&
      url.pathname === "/pull"
    ) {
      const body =
        await request.json();

      const instanceId =
        bodyValue(
          body,
          "instance_id"
        );

      const token =
        bodyValue(
          body,
          "token"
        );

      const identity =
        identities[instanceId];

      if (!identity) {
        return json(
          {
            error:
              "Unknown instance_id"
          },
          404
        );
      }

      if (
        identity.instance_token !==
        token
      ) {
        return unauthorized();
      }

      const messages =
        queues[instanceId] || [];

      queues[instanceId] = [];

      identity.last_seen =
        new Date().toISOString();

      await this.state.storage.put(
        "queues",
        queues
      );

      await this.state.storage.put(
        "identities",
        identities
      );

      return json({
        ok: true,
        instance_id:
          instanceId,
        identity:
          publicIdentity(identity),
        messages
      });
    }

    /*
     * Retrieve a known identity.
     */
    if (
      request.method === "GET" &&
      url.pathname === "/identity"
    ) {
      const instanceId =
        (
          url.searchParams.get(
            "instance_id"
          ) || ""
        ).trim();

      const identity =
        identities[instanceId];

      if (!identity) {
        return json(
          {
            error:
              "Unknown instance_id"
          },
          404
        );
      }

      return json({
        ok: true,
        identity:
          publicIdentity(identity)
      });
    }

    return json(
      {
        error: "Not found"
      },
      404
    );
  }
}


/*
 * Never return the master Nordic Hub secret.
 * The per-instance token is returned only when an
 * instance registers/reconnects.
 */
function publicIdentity(
  identity,
  includeToken = false
) {
  const result = {
    instance_id:
      identity.instance_id,
    name:
      identity.name,
    first_seen:
      identity.first_seen,
    last_seen:
      identity.last_seen,
    known:
      identity.known
  };

  if (includeToken) {
    result.instance_token =
      identity.instance_token;
  }

  return result;
}
```
