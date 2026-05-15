import tree_sitter_typescript as tsts
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

TS_LANGUAGE = Language(tsts.language_tsx())
JS_LANGUAGE = Language(tsjs.language())

def get_language(filename: str):
    if filename.endswith((".ts", ".tsx")):
        return TS_LANGUAGE
    return JS_LANGUAGE

def extract_chunks(filename: str, content: str) -> list[dict]:
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