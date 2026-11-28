"""将 docs/fer-animation/ 标准 Web 模块打包导出为单文件 docs/FER项目讲解动画.html。

用法（仅依赖 Python 3 标准库）：
    python docs/fer-animation/export.py                 导出单文件到 docs/FER项目讲解动画.html
    python docs/fer-animation/export.py --check         校验产物与源码是否一致（不一致退出码为 1）
    python docs/fer-animation/export.py --out PATH      导出到指定路径
    python docs/fer-animation/export.py --serve         启动本地预览服务器（自动修正 .js MIME）
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import re
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY_HTML = ROOT / "index.html"
DEFAULT_OUTPUT = ROOT.parent / "FER项目讲解动画.html"

LINK_RE = re.compile(r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*/?>')
SCRIPT_RE = re.compile(r'<script(?:\s+type="(module)")?\s+src="([^"]+)"\s*>\s*</script>')
LOCAL_ASSET_RE = re.compile(
    r'<(?:link[^>]+rel="stylesheet"[^>]+href|script[^>]+src)="(?!https?://|data:)([^"]+)"',
    re.I,
)


class ExportError(Exception):
    """源码结构或模块语法不符合导出约束。"""


def read_lf(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_$"


def skip_ws_and_comments(src: str, pos: int) -> int:
    n = len(src)
    while pos < n:
        ch = src[pos]
        if ch in " \t\r\n":
            pos += 1
        elif src.startswith("//", pos):
            nl = src.find("\n", pos + 2)
            pos = n if nl == -1 else nl + 1
        elif src.startswith("/*", pos):
            end = src.find("*/", pos + 2)
            if end == -1:
                raise ExportError("存在未闭合的块注释 /*")
            pos = end + 2
        else:
            break
    return pos


def find_import_end(src: str, pos: int) -> int:
    """从 import 关键字之后扫到分号或模块路径字符串后的换行。"""
    n = len(src)
    brace = 0
    seen_from_str = False
    mode = "code"
    while pos < n:
        ch = src[pos]
        if mode == "line_comment":
            if ch == "\n":
                mode = "code"
                if seen_from_str:
                    return pos
            pos += 1
            continue
        if mode == "block_comment":
            if src.startswith("*/", pos):
                mode = "code"
                pos += 2
            else:
                pos += 1
            continue
        if mode in ("'", '"'):
            if ch == "\\":
                pos += 2
                continue
            if ch == mode:
                mode = "code"
                seen_from_str = True
            pos += 1
            continue
        if src.startswith("//", pos):
            mode = "line_comment"
            pos += 2
            continue
        if src.startswith("/*", pos):
            mode = "block_comment"
            pos += 2
            continue
        if ch in ("'", '"'):
            mode = ch
            pos += 1
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == ";" and brace == 0:
            return pos + 1
        elif ch == "\n" and brace == 0 and seen_from_str:
            return pos
        pos += 1
    return n


def find_statement_end(src: str, pos: int) -> int:
    """从 export { ... } / export default 之后扫到分号或顶层闭合块。"""
    n = len(src)
    paren = 0
    bracket = 0
    brace = 0
    tpl_stack: list[int] = []
    mode = "code"

    while pos < n:
        ch = src[pos]
        if mode == "line_comment":
            if ch == "\n":
                mode = "code"
                if paren == 0 and bracket == 0 and brace == 0:
                    return pos
            pos += 1
            continue
        if mode == "block_comment":
            if src.startswith("*/", pos):
                mode = "code"
                pos += 2
            else:
                pos += 1
            continue
        if mode in ("'", '"'):
            if ch == "\\":
                pos += 2
                continue
            if ch == mode:
                mode = "code"
            pos += 1
            continue
        if mode == "`":
            if ch == "\\":
                pos += 2
                continue
            if src.startswith("${", pos):
                tpl_stack.append(brace)
                brace += 1
                mode = "code"
                pos += 2
                continue
            if ch == "`":
                mode = "code"
            pos += 1
            continue

        if src.startswith("//", pos):
            mode = "line_comment"
            pos += 2
            continue
        if src.startswith("/*", pos):
            mode = "block_comment"
            pos += 2
            continue
        if ch in ("'", '"', "`"):
            mode = ch
            pos += 1
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            if tpl_stack and brace - 1 == tpl_stack[-1]:
                tpl_stack.pop()
                brace -= 1
                mode = "`"
                pos += 1
                continue
            brace -= 1
        elif ch == ";" and paren == 0 and bracket == 0 and brace == 0:
            return pos + 1
        pos += 1
    return n


def scan_top_level_esm(src: str) -> list[tuple[int, int, str]]:
    """扫描顶层（brace/paren/bracket depth == 0）静态 import/export 语句范围。"""
    n = len(src)
    pos = 0
    paren = 0
    bracket = 0
    brace = 0
    tpl_stack: list[int] = []
    mode = "code"
    stmt_start_ready = True
    hits: list[tuple[int, int, str]] = []

    while pos < n:
        ch = src[pos]
        if mode == "line_comment":
            if ch == "\n":
                mode = "code"
            pos += 1
            continue
        if mode == "block_comment":
            if src.startswith("*/", pos):
                mode = "code"
                pos += 2
            else:
                pos += 1
            continue
        if mode in ("'", '"'):
            if ch == "\\":
                pos += 2
                continue
            if ch == mode:
                mode = "code"
            pos += 1
            continue
        if mode == "`":
            if ch == "\\":
                pos += 2
                continue
            if src.startswith("${", pos):
                tpl_stack.append(brace)
                brace += 1
                mode = "code"
                pos += 2
                continue
            if ch == "`":
                mode = "code"
            pos += 1
            continue

        if ch in " \t\r\n":
            pos += 1
            continue
        if src.startswith("//", pos):
            mode = "line_comment"
            pos += 2
            continue
        if src.startswith("/*", pos):
            mode = "block_comment"
            pos += 2
            continue
        if ch in ("'", '"', "`"):
            mode = ch
            stmt_start_ready = False
            pos += 1
            continue

        if paren == 0 and bracket == 0 and brace == 0 and stmt_start_ready:
            if src.startswith("import", pos):
                after = pos + 6
                if after < n and src[after] in " \t\r\n{*\"'":
                    end = find_import_end(src, after)
                    hits.append((pos, end, "import"))
                    pos = end
                    stmt_start_ready = True
                    continue
                if after < n and src[after] == "(":
                    raise ExportError("不支持动态 import()，请改用顶层静态 import")
            elif src.startswith("export", pos):
                after = pos + 6
                if after < n and src[after] in " \t\r\n{*":
                    rest_pos = skip_ws_and_comments(src, after)
                    if src.startswith("*", rest_pos) or (
                        src.startswith("{", rest_pos)
                        and re.match(r"export\s+\{[^}]*\}\s+from\b", src[pos : pos + 200])
                    ):
                        raise ExportError("不支持 re-export 语法")
                    if src.startswith("default", rest_pos) and (
                        rest_pos + 7 < n and not is_ident_char(src[rest_pos + 7])
                    ):
                        end = find_statement_end(src, rest_pos + 7)
                        hits.append((pos, end, "export_default"))
                        pos = end
                        stmt_start_ready = True
                        continue
                    if src.startswith("{", rest_pos):
                        end = find_statement_end(src, rest_pos)
                        hits.append((pos, end, "export_list"))
                        pos = end
                        stmt_start_ready = True
                        continue
                    m_decl = re.match(
                        r"export\s+(?:(?:const|let|var)\s+([A-Za-z_$][\w$]*)|(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*))(?![\w$])",
                        src[pos : pos + 120],
                    )
                    if not m_decl:
                        raise ExportError(f"不支持的 export 语法：{src[pos : pos + 60].splitlines()[0]}")
                    hits.append((pos, pos + m_decl.end(), "export_decl"))
                    pos = pos + m_decl.end()
                    stmt_start_ready = False
                    continue

        if ch == "(":
            paren += 1
            stmt_start_ready = False
        elif ch == ")":
            paren -= 1
            stmt_start_ready = False
        elif ch == "[":
            bracket += 1
            stmt_start_ready = False
        elif ch == "]":
            bracket -= 1
            stmt_start_ready = False
        elif ch == "{":
            brace += 1
            stmt_start_ready = False
        elif ch == "}":
            if tpl_stack and brace - 1 == tpl_stack[-1]:
                tpl_stack.pop()
                brace -= 1
                mode = "`"
                pos += 1
                continue
            brace -= 1
            stmt_start_ready = (paren == 0 and bracket == 0 and brace == 0)
        elif ch == ";":
            stmt_start_ready = (paren == 0 and bracket == 0 and brace == 0)
        else:
            stmt_start_ready = False
        pos += 1

    return hits


def resolve_module_path(importer: Path, spec: str) -> Path:
    if not (spec.startswith("./") or spec.startswith("../")):
        raise ExportError(f"{importer.relative_to(ROOT).as_posix()} 使用了非相对模块路径：{spec}")
    target = (importer.parent / spec).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ExportError(f"模块路径越界：{spec}") from exc
    if not target.is_file():
        raise ExportError(f"找不到依赖模块 {spec}（自 {importer.relative_to(ROOT).as_posix()}）")
    return target


def parse_named_specifiers(raw: str, is_import: bool) -> list[tuple[str, str]]:
    """解析 `{ a, b as c }`，返回 `[(orig, local_or_alias), ...]`。"""
    items = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        m = re.fullmatch(r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)", token)
        if m:
            items.append((m.group(1), m.group(2)))
        elif re.fullmatch(r"[A-Za-z_$][\w$]*", token):
            items.append((token, token))
        else:
            kind = "import" if is_import else "export"
            raise ExportError(f"无法解析的 {kind} 具名项：{token}")
    return items


def transform_module(mod_path: Path, src: str) -> tuple[list[Path], str]:
    """将单个 ESM 模块转为经典函数体源码，并返回依赖路径列表。"""
    hits = scan_top_level_esm(src)
    deps: list[Path] = []
    exports: list[tuple[str, str]] = []  # (exported_name, local_name)
    chunks: list[str] = []
    cursor = 0

    for start, end, kw in hits:
        chunks.append(src[cursor:start])
        stmt = src[start:end].strip()

        if kw == "import":
            m_side = re.fullmatch(r'import\s+["\']([^"\']+)["\']\s*;?', stmt)
            m_star = re.fullmatch(r'import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+["\']([^"\']+)["\']\s*;?', stmt)
            m_full = re.fullmatch(
                r'import\s+(.+?)\s+from\s+["\']([^"\']+)["\']\s*;?',
                stmt,
                re.S,
            )
            if m_side:
                dep = resolve_module_path(mod_path, m_side.group(1))
                deps.append(dep)
                replacement = ""
            elif m_star:
                ns, spec = m_star.group(1), m_star.group(2)
                dep = resolve_module_path(mod_path, spec)
                deps.append(dep)
                rel = dep.relative_to(ROOT).as_posix()
                replacement = f'const {ns} = __mods["{rel}"];'
            elif m_full:
                clause, spec = m_full.group(1).strip(), m_full.group(2)
                dep = resolve_module_path(mod_path, spec)
                deps.append(dep)
                rel = dep.relative_to(ROOT).as_posix()
                bindings: list[str] = []
                if "{" in clause:
                    brace_idx = clause.index("{")
                    head_def = clause[:brace_idx].strip().rstrip(",").strip()
                    if head_def:
                        bindings.append(f"default: {head_def}")
                    named_body = clause[brace_idx + 1 : clause.rindex("}")]
                    for orig, local in parse_named_specifiers(named_body, is_import=True):
                        bindings.append(orig if orig == local else f"{orig}: {local}")
                else:
                    bindings.append(f"default: {clause}")
                replacement = f'const {{ {", ".join(bindings)} }} = __mods["{rel}"];'
            else:
                raise ExportError(f"不支持的 import 语法：{stmt}")
            chunks.append(replacement)

        elif kw == "export_default":
            body = re.sub(r"^export\s+default\s+", "", stmt, count=1)
            chunks.append(f"const __default = {body}")
            exports.append(("default", "__default"))

        elif kw == "export_list":
            inner = stmt[stmt.index("{") + 1 : stmt.rindex("}")]
            for local, alias in parse_named_specifiers(inner, is_import=False):
                exports.append((alias, local))
            chunks.append("")

        elif kw == "export_decl":
            m_decl = re.match(
                r"^export\s+(?:(?:const|let|var)\s+([A-Za-z_$][\w$]*)|(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*))(?![\w$])",
                stmt,
            )
            if not m_decl:
                raise ExportError(f"不支持的 export 语法：{stmt}")
            name = m_decl.group(1) or m_decl.group(2) or m_decl.group(3)
            exports.append((name, name))
            chunks.append(re.sub(r"^export\s+", "", stmt, count=1))

        cursor = end

    chunks.append(src[cursor:])
    body = "".join(chunks).strip("\n")
    ret_items = ", ".join(loc if exp == loc else f"{exp}: {loc}" for exp, loc in exports)
    ret_line = f"\nreturn {{ {ret_items} }};" if exports else ""
    rel_self = mod_path.relative_to(ROOT).as_posix()
    wrapped = f'// --- {rel_self} ---\n__mods["{rel_self}"] = (() => {{\n{body}{ret_line}\n}})();'
    return deps, wrapped


def bundle_esm(entry_path: Path) -> tuple[str, int]:
    """从 ESM 入口出发做拓扑排序并打包为单个 IIFE。"""
    state: dict[Path, str] = {}  # "visiting" | "done"
    ordered_blocks: list[str] = []

    def dfs(mod: Path) -> None:
        st = state.get(mod)
        if st == "done":
            return
        if st == "visiting":
            raise ExportError(f"检测到模块循环依赖：{mod.relative_to(ROOT).as_posix()}")
        state[mod] = "visiting"
        deps, block = transform_module(mod, read_lf(mod))
        for dep in deps:
            dfs(dep)
        state[mod] = "done"
        ordered_blocks.append(block)

    dfs(entry_path.resolve())
    joined = "\n\n".join(ordered_blocks)
    bundle = f'(() => {{\n"use strict";\nconst __mods = {{}};\n\n{joined}\n}})();'
    return bundle, len(ordered_blocks)


def escape_inline_script(js: str) -> str:
    return js.replace("</script", "<\\/script")


def render_single_file() -> tuple[bytes, int]:
    if not ENTRY_HTML.is_file():
        raise ExportError(f"找不到入口文件：{ENTRY_HTML}")
    html = read_lf(ENTRY_HTML)
    mod_count = 0

    def inline_css(m: re.Match[str]) -> str:
        href = m.group(1)
        if href.startswith(("http://", "https://", "data:")):
            return m.group(0)
        css_path = (ROOT / href).resolve()
        if not css_path.is_file():
            raise ExportError(f"找不到样式文件：{href}")
        css = read_lf(css_path).strip("\n")
        if "</style" in css.lower():
            raise ExportError(f"样式文件包含非法闭合标签 </style：{href}")
        return f"<style>\n{css}\n</style>"

    def inline_js(m: re.Match[str]) -> str:
        nonlocal mod_count
        is_mod, src_attr = m.group(1), m.group(2)
        if src_attr.startswith(("http://", "https://", "data:")):
            return m.group(0)
        js_path = (ROOT / src_attr).resolve()
        if not js_path.is_file():
            raise ExportError(f"找不到脚本文件：{src_attr}")
        if is_mod == "module":
            bundled, count = bundle_esm(js_path)
            mod_count += count
            content = escape_inline_script(bundled)
        else:
            content = escape_inline_script(read_lf(js_path).strip("\n"))
        return f"<script>\n{content}\n</script>"

    html = LINK_RE.sub(inline_css, html)
    html = SCRIPT_RE.sub(inline_js, html)

    leftover = LOCAL_ASSET_RE.findall(html)
    if leftover:
        raise ExportError(f"导出结果仍残留未内联的本地资源：{leftover}")

    return (html.strip("\n") + "\n").encode("utf-8"), mod_count


def serve(port: int) -> int:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".js": "text/javascript; charset=utf-8",
            ".mjs": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
        }

    class ThreadingServer(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    handler = partial(QuietHandler, directory=str(ROOT))
    with ThreadingServer(("127.0.0.1", port), handler) as httpd:
        print(f"预览服务已启动：http://127.0.0.1:{port}/index.html （Ctrl+C 退出）")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n预览服务已停止。")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出或预览 FER 项目讲解动画")
    p.add_argument("--check", action="store_true", help="只校验导出结果与目标文件是否一致，不写文件")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="指定导出 HTML 文件路径")
    p.add_argument("--serve", action="store_true", help="启动本地静态预览服务")
    p.add_argument("--port", type=int, default=8000, help="--serve 监听端口（默认 8000）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        return serve(args.port)

    try:
        data, mod_count = render_single_file()
    except (ExportError, OSError, UnicodeDecodeError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 1

    out_path = args.out.resolve()
    digest = hashlib.sha256(data).hexdigest()[:16]

    if args.check:
        if not out_path.is_file() or out_path.read_bytes() != data:
            print(f"产物与源码不一致：{out_path}，请运行 python docs/fer-animation/export.py 重新导出")
            return 1
        print(f"产物与源码一致：{out_path.name}（{len(data)} 字节，{mod_count} 个模块，sha256 {digest}）")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"已导出 {out_path}（{len(data)} 字节，{mod_count} 个模块，sha256 {digest}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
