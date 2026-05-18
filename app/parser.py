import tree_sitter_typescript as tsts
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

TS_LANGUAGE = Language(tsts.language_tsx())
JS_LANGUAGE = Language(tsjs.language())

import re

def strip_comments(content: str, filename: str) -> str:
    if filename.endswith((".ts", ".tsx", ".js", ".jsx")):
        # remove single line comments
        content = re.sub(r"//.*", "", content)
        # remove multi line comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return content

def get_language(filename: str):
    if filename.endswith((".ts", ".tsx")):
        return TS_LANGUAGE
    return JS_LANGUAGE

def extract_chunks(filename: str, content: str) -> list[dict]:
    content = strip_comments(content, filename)
    lang = get_language(filename)
    parser = Parser(lang)
    tree = parser.parse(bytes(content, "utf-8"))
    root = tree.root_node

    chunks = []

    for node in root.children:
        if node.type in (
            "function_declaration",
            "export_statement",
            "class_declaration",
            "lexical_declaration",
        ):
            chunk_text = content[node.start_byte:node.end_byte]
            chunks.append({
                "filename": filename,
                "type": node.type,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "content": chunk_text
            })
            

    return chunks

def chunk_prd(content: str, chunk_size: int = 500) -> list[dict]:
    lines = content.split("\n")
    chunks = []
    current = []
    current_size = 0

    for line in lines:
        current.append(line)
        current_size += len(line)
        if current_size >= chunk_size and line.strip() == "":
            chunks.append({
                "content": "\n".join(current).strip(),
                "type": "prd_section",
                "filename": "PRD.md",
                "start_line": 0,
                "end_line": 0
            })
            current = []
            current_size = 0

    if current:
        chunks.append({
            "content": "\n".join(current).strip(),
            "type": "prd_section",
            "filename": "PRD.md",
            "start_line": 0,
            "end_line": 0
        })

    return chunks