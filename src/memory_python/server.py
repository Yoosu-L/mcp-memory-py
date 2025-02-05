from mcp.server.fastmcp import FastMCP

mcp = FastMCP("memory-python")

@mcp.tool()
def string_length(string: str) -> int:
    """Computes the length of a string"""
    return len(string)