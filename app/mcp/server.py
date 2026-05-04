"""
MCP (Model Context Protocol) Server.
Exposes the knowledge base as a tool that any MCP-compatible LLM client can call.

Run as standalone:  python -m app.mcp.server
This satisfies the manager's requirement to use MCP.
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from app.services.knowledge_base import kb
from app.services.ingestion import ingest_file


server = Server("kms-knowledge-base")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description=(
                "Search the VVDN internal knowledge base for relevant information. "
                "Returns the top-k most relevant text chunks from documents, video "
                "transcripts, and image descriptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_document",
            description="Add a file (PDF/image/video/text) to the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "source_name": {"type": "string"},
                },
                "required": ["file_path", "source_name"],
            },
        ),
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_knowledge_base":
        results = kb.search(arguments["query"], k=arguments.get("top_k", 5))
        if not results:
            return [TextContent(type="text", text="No relevant content found.")]
        formatted = "\n\n---\n\n".join(
            f"[{r['metadata'].get('type', 'doc')} | {r['metadata'].get('source', '?')}]\n{r['text']}"
            for r in results
        )
        return [TextContent(type="text", text=formatted)]

    if name == "ingest_document":
        result = ingest_file(arguments["file_path"], arguments["source_name"])
        return [TextContent(type="text", text=str(result))]

    if name == "kb_stats":
        return [TextContent(type="text", text=str(kb.stats()))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
