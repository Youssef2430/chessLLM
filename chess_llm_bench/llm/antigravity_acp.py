"""Minimal ACP client for the user's installed Antigravity subscription server."""

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

from .subscription import SubscriptionProvider, subscription_environment


def discover_installation():
    root = Path.home() / ".t3"
    binaries = sorted(
        (root / "tools/antigravity-acp").glob("*/versions/*/agy_acp_server.par")
    )
    profiles = sorted(
        (root / "userdata/providers/antigravity").glob(
            "*/antigravity-acp/settings.json"
        )
    )
    for profile in profiles:
        settings = json.loads(profile.read_text())
        if (
            settings.get("auth", {}).get("type") == "oauth-personal"
            and (profile.parent / "acp_token.json").exists()
        ):
            for binary in binaries:
                harness = binary.parent / "localharness_external"
                if harness.exists():
                    return binary, harness, profile.parent.parent
    return None


class ACPClient:
    def __init__(self):
        self.process = None
        self.serial = 0
        self.events = []
        self.tmp = None

    async def start(self):
        installation = discover_installation()
        if not installation:
            raise RuntimeError(
                "No installed Antigravity ACP server with personal subscription auth"
            )
        binary, harness, profile = installation
        self.tmp = tempfile.TemporaryDirectory(prefix="chess-acp-")
        env = subscription_environment()
        env.update(
            GEMINI_HOME=str(profile),
            AGY_ACP_FORCE_FILE_STORAGE="1",
            ANTIGRAVITY_HARNESS_PATH=str(harness),
            BROWSER="/usr/bin/false",
            PYTHONUNBUFFERED="1",
            TMPDIR=self.tmp.name,
        )
        self.process = await asyncio.create_subprocess_exec(
            str(binary),
            cwd=self.tmp.name,
            env=env,
            start_new_session=True,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=4 * 1024 * 1024,
        )
        return await self.rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "chess-benchmark", "version": "2"},
            },
        )

    async def rpc(self, method, params, timeout=120):
        self.serial += 1
        identifier = self.serial
        await self.send(
            {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
        )

        async def receive():
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Antigravity ACP server closed its output")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("id") == identifier and "method" not in event:
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    return event.get("result", {})
                if "method" in event and "id" in event:
                    if event["method"] == "session/request_permission":
                        result = {"outcome": {"outcome": "cancelled"}}
                        await self.send(
                            {"jsonrpc": "2.0", "id": event["id"], "result": result}
                        )
                    else:
                        await self.send(
                            {
                                "jsonrpc": "2.0",
                                "id": event["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Tools disabled for chess benchmark",
                                },
                            }
                        )
                self.events.append(event)

        return await asyncio.wait_for(receive(), timeout)

    async def send(self, event):
        self.process.stdin.write((json.dumps(event) + "\n").encode())
        await self.process.stdin.drain()

    async def new_session(self):
        return await self.rpc("session/new", {"cwd": self.tmp.name, "mcpServers": []})

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        if self.tmp:
            self.tmp.cleanup()


class AntigravityProvider(SubscriptionProvider):
    native_models = {
        "gemini-3.1-pro": "gemini-3.1-pro-low",
        "gemini-3.8-flash": "gemini-3.8-flash-low",
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.acp = ACPClient()

    @property
    def native_model(self):
        if self.spec.model == "gemini-3.1-pro":
            return {"low": "gemini-3.1-pro-low", "high": "gemini-pro-agent"}.get(
                self.reasoning_effort
            )
        if self.spec.model == "gemini-3.8-flash":
            return f"gemini-3.8-flash-{self.reasoning_effort}"
        return None

    async def transport(self, prompt, timeout_s):
        if self.native_model is None:
            raise RuntimeError(
                "Requested Antigravity model/effort combination is unavailable"
            )
        if not self.acp.process:
            await self.acp.start()
        self.acp.events = []
        session = await self.acp.new_session()
        sid = session["sessionId"]
        native = self.native_model
        available = {
            m["modelId"] for m in session.get("models", {}).get("availableModels", [])
        }
        if native not in available:
            raise RuntimeError(
                f"Requested model unavailable in ACP catalog: {self.spec.model}"
            )
        selected = await self.acp.rpc(
            "session/set_config_option",
            {"sessionId": sid, "configId": "model", "value": native},
        )
        selected_id = next(
            (
                c.get("currentValue")
                for c in selected.get("configOptions", [])
                if c.get("id") == "model"
            ),
            None,
        )
        if selected_id != native:
            raise RuntimeError(
                f"ACP did not confirm selected model {native}: {selected_id}"
            )
        result = await self.acp.rpc(
            "session/prompt",
            {
                "sessionId": sid,
                "prompt": [
                    {"type": "text", "text": self.system_prompt + "\n\n" + prompt}
                ],
            },
            timeout_s,
        )
        chunks, tool_calls, usage = [], [], {}
        for event in self.acp.events:
            update = event.get("params", {}).get("update", {})
            if update.get("sessionUpdate") == "agent_message_chunk":
                content = update.get("content", {})
                if content.get("type") == "text":
                    chunks.append(content.get("text", ""))
            if update.get("sessionUpdate") in {"tool_call", "tool_call_update"}:
                tool_calls.append(update)
            if update.get("sessionUpdate") == "usage_update":
                usage = update
        error = "Tool use invalidates unaided chess benchmark" if tool_calls else None
        if result.get("stopReason") != "end_turn":
            error = str(result)
        # Normalize only response transport. Preserve all original ACP events and
        # actual variant in the trace, with no invented token counts.
        normalized = [
            {
                "type": "system",
                "subtype": "init",
                "model": self.spec.model,
                "native_model": native,
            },
            {
                "type": "result",
                "result": error or "".join(chunks),
                "is_error": bool(error),
                "usage": {"acp_usage": usage},
            },
            {
                "type": "acp_trace",
                "native_model": native,
                "session_setup": session,
                "selection": selected,
                "events": self.acp.events,
                "result": result,
            },
        ]
        return 0, "\n".join(json.dumps(e) for e in normalized), ""

    async def close(self):
        await self.acp.close()


async def inspect():
    client = ACPClient()
    try:
        print(json.dumps(await client.start(), indent=2), flush=True)
        session = await client.new_session()
        print(
            json.dumps(
                {k: v for k, v in session.items() if k != "sessionId"}, indent=2
            ),
            flush=True,
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(inspect())
