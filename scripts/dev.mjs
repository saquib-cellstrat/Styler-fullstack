import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));

const commands = [
  {
    name: "backend",
    port: 8000,
    command: "uv",
    args: ["run", "fastapi", "dev", "app/main.py"],
    cwd: join(rootDir, "backend"),
  },
  {
    name: "frontend",
    port: 3000,
    command: "bun",
    args: ["dev"],
    cwd: join(rootDir, "frontend"),
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

const occupiedPorts = [];

for (const { name, port } of commands) {
  if (!(await isPortAvailable(port))) {
    occupiedPorts.push(`${name} port ${port}`);
  }
}

if (occupiedPorts.length > 0) {
  console.error(`Cannot start dev servers because ${occupiedPorts.join(" and ")} is already in use.`);
  console.error("Stop the existing server or free the port, then run `bun dev` again.");
  process.exit(1);
}

const children = commands.map(({ name, command, args, cwd }) => {
  console.log(`Starting ${name}...`);

  return spawn(command, args, {
    cwd,
    detached: process.platform !== "win32",
    stdio: "inherit",
  });
});

let shuttingDown = false;

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

for (const child of children) {
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
}

process.on("SIGINT", () => stopAll(0));
process.on("SIGTERM", () => stopAll(0));
