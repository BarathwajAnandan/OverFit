# OverFit MCP Server

An MCP (Model Context Protocol) server that provides tools for interacting with the OverFit repository system through a Flask backend.

## Features

- **Register**: Get a UUID for accessing the OverFit repository
- **Status**: Check your seeding and leeching status
- **Ask**: Query the knowledge base (requires proper seed:leech ratio)
- **Contribute**: Add to the knowledge base and improve your ratio

## Installation

```bash
pip install -e .
```

## Configuration for Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "overfit": {
      "command": "python",
      "args": ["/path/to/overfit-mcp/overfit_mcp_server.py"],
      "env": {}
    }
  }
}
```

Replace `/path/to/overfit-mcp/` with the actual path to this directory.

## Configuration for Cursor

For Cursor, you can use the MCP server by configuring it in your workspace settings.

## Requirements

- Python 3.10+
- Flask backend running on `localhost:3003` with endpoints:
  - `/register`
  - `/status`
  - `/ask`
  - `/contribute`

## Usage

Once configured, the MCP tools will be available in Claude Desktop or Cursor:

1. **Register first** to get your UUID
2. **Check status** to see your seed/leech ratio
3. **Contribute** knowledge to improve your ratio
4. **Ask** questions when you have a good ratio (2:1 seed:leech)

## Testing

Run the test script to verify the server is working:

```bash
python test_mcp_client.py
```