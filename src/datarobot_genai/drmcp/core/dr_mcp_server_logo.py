# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

from fastmcp import FastMCP
from rich.align import Align
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from datarobot_genai import __version__ as datarobot_genai_version


# Green color #81FBA5 = RGB(129, 251, 165)
def _apply_green(text: str) -> str:
    """Apply green color #81FBA5 to all characters in the text."""
    # Apply ANSI escape code for RGB color #81FBA5 (129, 251, 165)
    green_start = "\x1b[38;2;129;251;165m"
    green_end = "\x1b[39m"
    # Wrap the entire text (except trailing newline) with green color codes
    lines = text.split("\n")
    colored_lines = [green_start + line + green_end if line else "" for line in lines]
    return "\n".join(colored_lines)


DR_LOGO_ASCII = _apply_green(r"""
 ____        _        ____       _           _   
|  _ \  __ _| |_ __ _|  _ \ ___ | |__   ___ | |_ 
| | | |/ _` | __/ _` | |_) / _ \| '_ \ / _ \| __|
| |_| | (_| | || (_| |  _ < (_) | |_) | (_) | |_ 
|____/ \__,_|\__\__,_|_| \_\___/|_.__/ \___/ \__|
""")


def log_server_custom_banner(
    server: FastMCP[Any],
    transport: str,
    *,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    tools_count: int | None = None,
    prompts_count: int | None = None,
    resources_count: int | None = None,
) -> None:
    """
    Create and log a formatted banner with server information and logo.

    Args:
        transport: The transport protocol being used
        server_name: Optional server name to display
        host: Host address (for HTTP transports)
        port: Port number (for HTTP transports)
        path: Server path (for HTTP transports)
        tools_count: Number of tools registered
        prompts_count: Number of prompts registered
        resources_count: Number of resources registered
    """
    # Create the logo text
    # Use Text with no_wrap and markup disabled to preserve ANSI escape codes
    logo_text = Text.from_ansi(DR_LOGO_ASCII, no_wrap=True)

    # Create the main title
    title_text = Text(f"DataRobot MCP Server {datarobot_genai_version}", style="dim green")
    stats_text = Text(
        f"{tools_count} tools, {prompts_count} prompts, {resources_count} resources",
        style="bold green",
    )

    # Create the information table
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold", justify="center")  # Emoji column
    info_table.add_column(style="cyan", justify="left")  # Label column
    info_table.add_column(style="dim", justify="left")  # Value column

    match transport:
        case "http" | "streamable-http":
            display_transport = "HTTP"
        case "sse":
            display_transport = "SSE"
        case "stdio":
            display_transport = "STDIO"

    info_table.add_row("🖥", "Server name:", Text(server.name + "\n", style="bold blue"))
    info_table.add_row("📦", "Transport:", display_transport)
    info_table.add_row("🌐", "MCP port:", str(port))

    # Show connection info based on transport
    if transport in ("http", "streamable-http", "sse") and host and port:
        server_url = f"http://{host}:{port}"
        if path:
            server_url += f"/{path.lstrip('/')}"
        info_table.add_row("🔗", "Server URL:", server_url)

    # Add documentation link
    info_table.add_row("", "", "")
    info_table.add_row("📚", "Docs:", "https://docs.datarobot.com")
    info_table.add_row("🚀", "Hosting:", "https://datarobot.com")

    # Create panel with logo, title, and information using Group
    panel_content = Group(
        Align.center(logo_text),
        "",
        Align.center(title_text),
        Align.center(stats_text),
        "",
        "",
        Align.center(info_table),
    )

    panel = Panel(
        panel_content,
        border_style="dim",
        padding=(1, 4),
        # expand=False,
        width=80,  # Set max width for the pane
    )

    console = Console(stderr=True)
    # Center the panel itself
    console.print(Group("\n", Align.center(panel), "\n"))
