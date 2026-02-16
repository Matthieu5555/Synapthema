/**
 * Mermaid diagram syntax validator.
 *
 * Sets up a minimal DOM environment (jsdom + isomorphic-dompurify) so that
 * mermaid's parse() function works in Node.js.
 *
 * Reads mermaid diagram code from stdin and attempts to parse it.
 * Exit code 0 = valid syntax, exit code 1 = invalid (error on stderr).
 *
 * Usage:
 *   echo "flowchart TD\n  A --> B" | node src/rendering/mermaid_validate.mjs
 */

import { JSDOM } from "jsdom";
import DOMPurify from "isomorphic-dompurify";

// Set up minimal browser globals that mermaid expects
const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", {
  value: dom.window.navigator,
  writable: true,
  configurable: true,
});
globalThis.DOMPurify = DOMPurify;
if (!globalThis.SVGElement) {
  globalThis.SVGElement = dom.window.SVGElement || class SVGElement {};
}

// Import mermaid after globals are set up
const mermaid = (await import("mermaid")).default;

mermaid.initialize({ startOnLoad: false });

let input = "";
for await (const chunk of process.stdin) {
  input += chunk;
}

input = input.trim();
if (!input) {
  process.stderr.write("Empty input\n");
  process.exit(1);
}

try {
  await mermaid.parse(input);
  process.exit(0);
} catch (err) {
  const msg = err.message || err.str || String(err);
  process.stderr.write(msg + "\n");
  process.exit(1);
}
