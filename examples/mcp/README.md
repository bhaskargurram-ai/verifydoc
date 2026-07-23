# Use VerifyDoc from any MCP-capable agent / IDE

VerifyDoc ships an MCP (Model Context Protocol) server that exposes a
document-extraction **trust layer** to agents. It runs over stdio — no ports, no
cloud. Install once:

```bash
pip install 'verifydoc[mcp]'   # provides the `verifydoc-mcp` command
```

The server exposes two tools:
- `verify_extraction(document, schema, threshold?, k?, adapter?)` → per-field `{value, confidence, grounding, decision}`
- `list_adapters()` → available extractor adapters

Point your client at `verifydoc-mcp`. Copy-paste the block for your tool:

### Claude Code
```bash
claude mcp add verifydoc -- verifydoc-mcp
```
(or add it to `.mcp.json` in your project — see below). The bundled skill in
`.claude/skills/verifydoc/SKILL.md` teaches Claude Code when to call it.

### Claude Desktop — `claude_desktop_config.json`
```json
{
  "mcpServers": {
    "verifydoc": { "command": "verifydoc-mcp" }
  }
}
```

### Cursor — `.cursor/mcp.json`
```json
{
  "mcpServers": {
    "verifydoc": { "command": "verifydoc-mcp" }
  }
}
```

### Cline (VS Code) — `cline_mcp_settings.json`
```json
{
  "mcpServers": {
    "verifydoc": { "command": "verifydoc-mcp", "disabled": false }
  }
}
```

### OpenAI Codex / any stdio-MCP client — generic `.mcp.json`
```json
{
  "mcpServers": {
    "verifydoc": { "command": "verifydoc-mcp", "args": [] }
  }
}
```

## Privacy note
By default the server uses the local `text-search` adapter (nothing leaves the
machine). Pass `adapter: "rapidocr"` for local OCR, or `adapter: "api-vlm"` only
if you explicitly want a hosted model to see the document.
