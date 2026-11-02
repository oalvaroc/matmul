-- Pandoc filter: replaces
--
--   ```{.include file="build/tab_blocos.md"}
--   Caption, in Markdown.
--   ```
--
-- by the table in the file, written by analysis.py, with that caption.

function CodeBlock(el)
  if not el.classes:includes("include") then return nil end
  local f = assert(io.open(el.attributes.file, "r"))
  local text = f:read("a")
  f:close()
  if el.text ~= "" and text:sub(1, 1) == "|" then
    text = text .. "\n: " .. el.text:gsub("\n", " ") .. "\n"
  end
  return pandoc.read(text, "markdown", {columns = 1000}).blocks
end
