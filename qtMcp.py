# nt6mcp.py - MCP server that forwards to nt6diffusion HTTP API - qtMcp
import os
import sys
import json
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Config: endpoints served by nt6diffusion.py
MAX_PAGES = 5
NT6DIFF_URL = os.environ.get("NT6DIFF_URL", "http://127.0.0.1:5015/generate_storybook")
NT6DIFF_SWITCH_URL = os.environ.get("NT6DIFF_SWITCH_URL", "http://127.0.0.1:5015/switch_character")

# Use stdlib to avoid extra deps
import urllib.request
import urllib.error

mcp = FastMCP(name="NT6 Diffusion MCP")

def _normalize_page(p: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # Accept {tokens, story_text} or map from {title, text}
    if "story_text" not in p:
        if "title" in p and "text" in p:
            p["story_text"] = f"{p['title']}\n\n{p['text']}"
        elif "text" in p:
            p["story_text"] = p["text"]
        else:
            raise ValueError(f"Page {idx}: missing story_text")
    if "tokens" not in p:
        if "title" in p:
            p["tokens"] = p["title"].lower().split()
        else:
            raise ValueError(f"Page {idx}: missing tokens")
    if not isinstance(p["tokens"], list) or not all(isinstance(t, str) for t in p["tokens"]):
        raise ValueError(f"Page {idx}: tokens must be list[str]")
    if not isinstance(p["story_text"], str):
        raise ValueError(f"Page {idx}: story_text must be str")
    return {"tokens": p["tokens"], "story_text": p["story_text"]}

@mcp.tool()
def generate_storybook(pages: List[Dict[str, Any]], save: bool = True) -> str:
    """
    Queue 1–5 storybook pages for generation by nt6diffusion.py.

    Arguments:
    - pages: list[{tokens:list[str], story_text:str}] or {title,text} mapped to {tokens,story_text}
    - save: forwarded flag (currently ignored by server but accepted)

    Behavior:
    - Returns an acknowledgment immediately. Images are saved by the diffusion service.
    """
    if not isinstance(pages, list) or not (1 <= len(pages) <= MAX_PAGES):
        return f"Error: 'pages' must be a list of 1..{MAX_PAGES} items."
    try:
        norm = [_normalize_page(p, idx) for idx, p in enumerate(pages, start=1)]
    except ValueError as e:
        return f"Error: {e}"

    payload = json.dumps({"pages": norm, "save": bool(save)}).encode("utf-8")
    req = urllib.request.Request(
        NT6DIFF_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return f"Error: diffusion service not reachable at {NT6DIFF_URL}. Launch nt6diffusion.py first. Details: {e}"
    except Exception as e:
        return f"Error: forwarding failed: {e}"

    if status >= 400:
        return f"Error: diffusion service responded {status}: {body}"

    return f"Accepted {len(norm)} page(s) for generation."

@mcp.tool()
def switch_character(tokens: List[str], reference_image_path: str = "", weight: float = 0.6, seed: int = None) -> str:
    """
    Set an active character for subsequent pages.

    Arguments:
    - tokens: list[str] — descriptors (e.g., ["young boy", "red hat"])
    - reference_image_path: optional local path to a reference image
    - weight: IP-Adapter strength 0..1
    - seed: optional int for reproducibility when text-generating the reference
    """
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t.strip() for t in tokens):
        return "Error: 'tokens' must be a non-empty list[str]."
    try:
        weight = float(weight)
    except Exception:
        return "Error: 'weight' must be a number."

    payload_dict = {"tokens": tokens, "reference_image_path": reference_image_path or None, "weight": weight}
    if isinstance(seed, int):
        payload_dict["seed"] = seed

    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        NT6DIFF_SWITCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return f"Error: diffusion service not reachable at {NT6DIFF_SWITCH_URL}. Launch nt6diffusion.py first. Details: {e}"
    except Exception as e:
        return f"Error: forwarding failed: {e}"

    if status >= 400:
        return f"Error: diffusion service responded {status}: {body}"

    return "Character switched successfully."

if __name__ == "__main__":
    mcp.run()