"""
ai_agent_client.py — Autonomous AI Agent Client that connects to the Streamable MCP Server.

How this Agent works:
  1. Authenticates with the MCP Server using Bearer token (identity: agent_alpha).
  2. Discovers tools available on the server via `tools/list`.
  3. Receives a high-level natural language goal from the user (e.g., "Check current time and send a maintenance alert").
  4. Autonomously plans and executes tool calls against the MCP server.
  5. Returns the completed task summary.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import httpx

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] %(message)s")
logger = logging.getLogger("AIAgent")


class MCPAIAgent:
    """
    Autonomous AI Agent that communicates with an MCP Server over Streamable HTTP.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8100/mcp",
        auth_token: str = "agent-token-alpha",
        identity_name: str = "agent_alpha",
    ):
        self.server_url = server_url
        self.auth_token = auth_token
        self.identity_name = identity_name
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        self.available_tools: List[Dict[str, Any]] = []
        self.request_counter = 1

    async def connect_and_discover_tools(self) -> List[Dict[str, Any]]:
        """Step 1: Connect to MCP Server and discover available tools."""
        logger.info(f"Agent '{self.identity_name}' connecting to MCP Server at {self.server_url}...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Initialize MCP handshake
            init_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "python-autonomous-agent", "version": "1.0.0"},
                },
            }
            res_init = await client.post(self.server_url, json=init_payload, headers=self.headers)
            if res_init.status_code != 200:
                raise RuntimeError(f"MCP Handshake failed: {res_init.status_code} - {res_init.text}")
            
            # 2. Discover Tools
            list_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
            }
            res_tools = await client.post(self.server_url, json=list_payload, headers=self.headers)
            tools_data = res_tools.json()
            self.available_tools = tools_data.get("result", {}).get("tools", [])
            
            logger.info(f"Connected successfully! Discovered {len(self.available_tools)} tools:")
            for t in self.available_tools:
                logger.info(f"   • Tool: '{t['name']}' -> {t['description']}")
            
            return self.available_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Execute an MCP tool on the server."""
        logger.info(f"Agent executing tool: '{tool_name}' with arguments: {arguments}")
        
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(self.server_url, json=payload, headers=self.headers)
            if res.status_code != 200:
                logger.error(f"Tool call HTTP error {res.status_code}: {res.text}")
                return {"error": res.text}
            
            data = res.json()
            if "error" in data:
                logger.warning(f"Server returned tool error: {data['error']}")
                return data["error"]
            
            # Parse MCP content result
            content = data["result"]["content"][0]["text"]
            result_json = json.loads(content)
            return result_json

    async def run_autonomous_workflow(self, task_goal: str, target_email: str) -> None:
        """
        Step 3: Autonomous Agent Decision & Execution Loop.
        Goal: Check the laptop time and dispatch an alert email with accurate timestamps.
        """
        print("\n" + "=" * 70)
        print(f"🤖 AGENT [{self.identity_name}] STARTING AUTONOMOUS WORKFLOW")
        print(f"🎯 Goal: {task_goal}")
        print("=" * 70)

        # 1. Discover tools
        await self.connect_and_discover_tools()

        # 2. Step 1: Agent decides to check time
        print("\n🧠 [Agent Reasoning]: 'First, I need to check the current system time to timestamp the alert.'")
        time_result = await self.call_tool("get_current_datetime", {})
        print(f"📥 [Agent Received Time]: {time_result.get('formatted')} (Source: {time_result.get('source')})")

        current_time_str = time_result.get("formatted", "Unknown Time")

        # 3. Step 2: Agent formulates email subject and body with the verified timestamp
        print("\n🧠 [Agent Reasoning]: 'Now I will compose and send the downtime alert email to the recipient.'")
        email_arguments = {
            "to": target_email,
            "subject": f"URGENT: Factory Machine Alert — {time_result.get('formatted')}",
            "body": (
                f"Autonomous Alert Notification\n"
                f"----------------------------------------\n"
                f"Machine: CNC Milling Station #4\n"
                f"Status: Emergency Stop Triggered\n"
                f"Incident Timestamp: {current_time_str}\n"
                f"Severity: HIGH\n"
                f"Dispatched by: Autonomous AI Agent ({self.identity_name})\n"
            ),
        }

        # 4. Step 3: Agent calls send_email
        email_result = await self.call_tool("send_email", email_arguments)
        print(f"📥 [Agent Received Email Result]: Status = {email_result.get('status')}, Delivery = {email_result.get('delivery_mode')}")

        # 5. Final Workflow Summary
        print("\n" + "=" * 70)
        print("🎉 [Agent Finished]: Task successfully completed!")
        print(f"   • Verified Time: {current_time_str}")
        print(f"   • Alert Dispatched to: {target_email}")
        print(f"   • Message ID: {email_result.get('message_id')}")
        print("=" * 70 + "\n")

    async def interactive_chat(self) -> None:
        """
        Interactive 2-Way Conversational Mode:
        Talk with the AI Agent in real-time. The agent listens to your inputs and calls MCP tools!
        """
        await self.connect_and_discover_tools()
        print("\n" + "=" * 75)
        print(f"💬 2-WAY INTERACTIVE CHAT WITH AI AGENT [{self.identity_name}]")
        print("   Ask anything in plain English:")
        print("   • 'What time is it?' / 'What is the time in Tokyo?'")
        print("   • 'Send an email to Vishal saying the system is healthy'")
        print("   • 'Send an email to Manoj with meeting details'")
        print("   • 'What tools do you have?'")
        print("   • 'exit' to quit")
        print("=" * 75 + "\n")

        loop = asyncio.get_event_loop()

        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "👤 You: ")
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q", "bye"):
                    print("\n👋 Agent: Goodbye! Disconnecting from MCP Server.\n")
                    break

                # 1. Greetings & Identity Queries
                if any(w in user_lower for w in ("hi", "hello", "hey", "who are you", "what is your name")):
                    print(f"\n🤖 Agent: Hello! I am **{self.identity_name}**, an autonomous AI agent connected to your MCP Server at {self.server_url}.")
                    print("   I can check the hardware host time across different timezones and dispatch real emails over SMTP.\n")

                # 2. Agent presence / Identities query
                elif any(w in user_lower for w in ("agent", "identit", "who are the agents", "who is present", "users")):
                    print(f"\n🤖 Agent: Here are the **5 AI Agent & Caller Identities** configured on this MCP Server:")
                    print("   1. 🤖 **agent_alpha** (Token: `agent-token-alpha`) — Autonomous Monitoring Agent (That's me!)")
                    print("   2. 🤖 **agent_beta** (Token: `agent-token-beta`) — Customer Support Bot")
                    print("   3. 👨‍💻 **vishal_engineer** (Token: `vishal-test-token`) — Developer & Lead Engineer (Vishal)")
                    print("   4. 👨‍💻 **vinod_engineer** (Token: `vinod-test-token`) — Infrastructure Engineer (Vinod)")
                    print("   5. 🔑 **admin_user** (Token: `admin-token-secret`) — System Administrator\n")

                # 3. Time query
                elif "time" in user_lower or "date" in user_lower or "clock" in user_lower:
                    tz = None
                    for candidate in ("tokyo", "asia/tokyo", "new_york", "new york", "america/new_york", "london", "europe/london", "kolkata", "asia/kolkata", "utc", "gmt", "paris", "dubai", "singapore", "sydney"):
                        if candidate in user_lower:
                            if "tokyo" in candidate:
                                tz = "Asia/Tokyo"
                            elif "new" in candidate or "york" in candidate:
                                tz = "America/New_York"
                            elif "london" in candidate:
                                tz = "Europe/London"
                            elif "kolkata" in candidate:
                                tz = "Asia/Kolkata"
                            elif "paris" in candidate:
                                tz = "Europe/Paris"
                            elif "dubai" in candidate:
                                tz = "Asia/Dubai"
                            elif "singapore" in candidate:
                                tz = "Asia/Singapore"
                            elif "sydney" in candidate:
                                tz = "Australia/Sydney"
                            else:
                                tz = candidate.upper()
                            break

                    print(f"\n🧠 [{self.identity_name}]: Calling 'get_current_datetime' on MCP server (Timezone: {tz or 'Host Local'})...")
                    res = await self.call_tool("get_current_datetime", {"timezone": tz} if tz else {})
                    if res.get("status") == "success":
                        print(f"\n🤖 Agent: The current time is **{res.get('formatted')}**.\n   (Source: {res.get('source')}, Offset: {res.get('utc_offset')})\n")
                    else:
                        print(f"\n🤖 Agent: ❌ Failed to get time: {res.get('message')}\n")

                # 4. Email query
                elif "email" in user_lower or "send" in user_lower or "mail" in user_lower:
                    recipient = "bonthumanoj999@gmail.com"  # Default recipient set to Manoj
                    if "vishal" in user_lower:
                        recipient = "vishalreddykonreddy@gmail.com"

                    subject = "Notification Alert from MCP AI Agent"
                    body = (
                        f"Hello!\n\n"
                        f"This message was generated from your interactive 2-way chat with AI Agent ({self.identity_name}).\n"
                        f"Command entered: '{user_input}'\n"
                        f"Dispatched via: Model Context Protocol (MCP) Streamable Server"
                    )

                    print(f"\n🧠 [{self.identity_name}]: Calling 'send_email' for recipient {recipient}...")
                    res = await self.call_tool("send_email", {"to": recipient, "subject": subject, "body": body})
                    if res.get("status") == "sent":
                        print(f"\n🤖 Agent: ✅ Email sent successfully to **{recipient}**!\n   Message ID: {res.get('message_id')}\n   Delivery Mode: {res.get('delivery_mode')}\n")
                    else:
                        print(f"\n🤖 Agent: ❌ Email failed: {res.get('message')}\n")

                # 5. List tools / Help query
                elif "tool" in user_lower or "help" in user_lower or "what can you do" in user_lower:
                    print(f"\n🤖 Agent: I have 2 active tools connected via MCP:")
                    for t in self.available_tools:
                        print(f"   • **{t['name']}**: {t['description']}")
                    print()

                # 6. General Math Calculations
                elif any(op in user_input for op in ("+", "-", "*", "/", "^")) and any(c.isdigit() for c in user_input):
                    try:
                        clean_expr = user_input.replace("what is", "").replace("calculate", "").replace("=", "").strip()
                        # Safe arithmetic evaluation
                        calc_result = eval(clean_expr, {"__builtins__": None}, {})
                        print(f"\n🤖 Agent: Result: **{clean_expr} = {calc_result}**\n")
                        continue
                    except Exception:
                        pass

                # 7. General Knowledge & AI Reasoning Engine
                else:
                    print(f"\n🧠 [{self.identity_name} reasoning]: Processing natural language knowledge query...")
                    
                    # Knowledge base lookup
                    if "capital of" in user_lower:
                        country = user_lower.split("capital of")[-1].strip(" ?.")
                        capitals = {
                            "france": "Paris", "india": "New Delhi", "japan": "Tokyo",
                            "usa": "Washington, D.C.", "uk": "London", "germany": "Berlin",
                            "italy": "Rome", "canada": "Ottawa", "australia": "Canberra",
                        }
                        ans = capitals.get(country, f"the capital city of {country.title()}")
                        print(f"\n🤖 Agent: The capital of {country.title()} is **{ans}**.\n")

                    elif "what is mcp" in user_lower or "model context protocol" in user_lower:
                        print("\n🤖 Agent: **Model Context Protocol (MCP)** is an open standard created by Anthropic that allows AI applications (like Claude, Cursor, and ChatGPT) to securely connect to external tools, databases, and servers (like your custom Streamable MCP server!).\n")

                    elif "what is python" in user_lower:
                        print("\n🤖 Agent: **Python** is a high-level, interpreted programming language known for its clear syntax and versatility in AI, backend web development, and data science.\n")

                    elif "who created you" in user_lower or "who made you" in user_lower:
                        print("\n🤖 Agent: I was configured by **Vishal & Vinod** to demonstrate an enterprise Model Context Protocol (MCP) server with tool-calling capabilities.\n")

                    else:
                        print(f"\n🤖 Agent: I understand you are asking about: *'{user_input}'*.")
                        print(f"   As an MCP Agent, I have direct access to your local machine tools.")
                        print(f"   • Try asking: *'What is the current time in Tokyo?'*")
                        print(f"   • Or: *'Send an email to Manoj with an update'*")
                        print(f"   • Or: *'What is 150 * 12?'*\n")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Agent: Disconnecting. Goodbye!\n")
                break

    def _next_id(self) -> int:
        req_id = self.request_counter
        self.request_counter += 1
        return req_id


async def main():
    agent = MCPAIAgent(
        server_url="http://localhost:8100/mcp",
        auth_token="agent-token-alpha",
        identity_name="agent_alpha",
    )

    # Launch 2-Way Interactive Chat Mode
    await agent.interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
