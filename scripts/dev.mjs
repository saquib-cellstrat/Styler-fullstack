import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const backendUrl = "http://127.0.0.1:8000";

const commands = [
  {
    name: "backend",
    port: 8000,
    command: "uv",
    args: ["run", "fastapi", "dev", "app/main.py"],
    cwd: join(rootDir, "backend"),
    readyUrl: `${backendUrl}/docs`,
  },
  {
    name: "frontend",
    port: 3000,
    command: "bun",
    args: ["dev"],
    cwd: join(rootDir, "frontend"),
    env: {
      HAIRSWAP_BACKEND_URL: backendUrl,
    },
  },
];

function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = createServer()
      .once("error", () => resolve(false))
      .once("listening", () => {
        server.close(() => resolve(true));
      })
      .listen(port, "127.0.0.1");
  });
}

async function findAvailablePort(preferredPort, attempts = 20) {
  for (let offset = 0; offset < attempts; offset += 1) {
    const port = preferredPort + offset;
    if (await isPortAvailable(port)) {
      return port;
    }
  }

  return null;
}

async function canFetch(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2000);

  try {
    const response = await fetch(url, {
      cache: "no-store",
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function getExistingFrontend(lockPath, preferredPort) {
  if (!existsSync(lockPath)) {
    return null;
  }

  try {
    const lock = JSON.parse(readFileSync(lockPath, "utf8"));
    if (
      lock &&
      lock.port === preferredPort &&
      typeof lock.pid === "number"
    ) {
      return lock;
    }
  } catch {
    return null;
  }

  return null;
}

const occupiedPorts = [];
const [backend, frontend] = commands;
const existingFrontend = getExistingFrontend(
  join(frontend.cwd, ".next", "dev", "lock"),
  frontend.port,
);

const backendPortAvailable = await isPortAvailable(backend.port);
if (!backendPortAvailable) {
  if (await canFetch(backend.readyUrl)) {
    backend.reuseUrl = backendUrl;
  } else {
    occupiedPorts.push(`${backend.name} port ${backend.port}`);
  }
}

const preferredFrontendPortAvailable = await isPortAvailable(frontend.port);

if (!preferredFrontendPortAvailable && existingFrontend) {
  frontend.reuseUrl =
    typeof existingFrontend.appUrl === "string"
      ? existingFrontend.appUrl
      : `http://127.0.0.1:${frontend.port}`;
} else {
  const frontendPort = await findAvailablePort(frontend.port);
  if (frontendPort === null) {
    occupiedPorts.push(`${frontend.name} ports ${frontend.port}-${frontend.port + 19}`);
  } else {
    frontend.port = frontendPort;
    frontend.args = [...frontend.args, "--port", String(frontendPort)];
  }

  if (frontendPort !== null && frontendPort !== 3000) {
    console.log(`Frontend port 3000 is in use; using ${frontendPort} instead.`);
  }
}

if (occupiedPorts.length > 0) {
  console.error(`Cannot start dev servers because ${occupiedPorts.join(" and ")} is already in use.`);
  console.error("Stop the existing server or free the port, then run `bun dev` again.");
  process.exit(1);
}

const children = [];
let shuttingDown = false;

function startCommand({ name, command, args, cwd, env = {} }) {
  console.log(`Starting ${name}...`);

  const child = spawn(command, args, {
    cwd,
    detached: process.platform !== "win32",
    env: {
      ...process.env,
      ...env,
    },
    stdio: "inherit",
  });

  children.push(child);

  child.on("error", (error) => {
    console.error(error.message);
    stopAll(1);
  });

  child.on("exit", (code, signal) => {
    if (!shuttingDown) {
      console.log(`A dev server stopped (${signal ?? `exit ${code ?? 0}`}).`);
      stopAll(code ?? 1);
    }
  });

  return child;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp({ name, readyUrl }, child) {
  if (!readyUrl) {
    return;
  }

  const timeoutMs = 120_000;
  const intervalMs = 500;
  const deadline = Date.now() + timeoutMs;
  console.log(`Waiting for ${name} at ${readyUrl}...`);

  while (Date.now() < deadline) {
    if (shuttingDown) {
      return;
    }
    if (child.exitCode !== null) {
      throw new Error(`${name} stopped before it was ready.`);
    }
    if (await canFetch(readyUrl)) {
      console.log(`${name} is ready.`);
      return;
    }

    await sleep(intervalMs);
  }

  throw new Error(`${name} did not become ready within ${timeoutMs / 1000}s.`);
}

function stopAll(exitCode = 0) {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;
  process.exitCode = exitCode;

  for (const child of children) {
    if (!child.pid || child.exitCode !== null) {
      continue;
    }

    try {
      if (process.platform === "win32") {
        child.kill("SIGTERM");
      } else {
        process.kill(-child.pid, "SIGTERM");
      }
    } catch {
      // The process may already have exited.
    }
  }

  setTimeout(() => process.exit(exitCode), 500);
}

if (backend.reuseUrl) {
  console.log(`Using existing backend at ${backend.reuseUrl}.`);
} else {
  const backendChild = startCommand(backend);

  try {
    await waitForHttp(backend, backendChild);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    stopAll(1);
  }
}

if (!shuttingDown) {
  if (frontend.reuseUrl) {
    console.log(`Using existing frontend at ${frontend.reuseUrl}.`);
  } else {
    startCommand(frontend);
  }
}

process.on("SIGINT", () => stopAll(0));
process.on("SIGTERM", () => stopAll(0));
