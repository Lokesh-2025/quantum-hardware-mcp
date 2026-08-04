"""
Shared MCP server instance.
===========================
server.py (IBM + IonQ hardware tools) and tools_chemistry.py (qforge chemistry
tools) both need to hang @mcp.tool() decorators off the SAME FastMCP object.

If either module owned it, the other would have to import that module. Since
server.py imports tools_chemistry to register its tools, that would be a
circular import. Parking the instance here means both sides import from a
module that depends on neither.

The name "quantum-hardware" is what Claude Desktop shows in its UI.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("quantum-hardware")
