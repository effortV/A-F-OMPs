"""描述符代码沙箱：AST 静态白名单检查 + 子进程执行 + 超时。

发现层的 Agent 会自己写 Python 计算新描述符。这里的约束是硬的：
- 只允许 import 白名单模块（numpy / math / rdkit / statistics / itertools / functools）
- 禁止 open / exec / eval / __import__ / os / sys / subprocess / socket / requests
- 子进程执行，超时强杀，无返回值即判失败
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..core.config import CFG

ALLOWED_MODULES = {
    "numpy", "np", "math", "statistics", "itertools", "functools", "collections",
    "rdkit", "rdkit.Chem", "rdkit.Chem.AllChem", "rdkit.Chem.Descriptors",
    "rdkit.Chem.rdMolDescriptors", "rdkit.Chem.Descriptors3D", "rdkit.Chem.Crippen",
    "rdkit.Chem.rdFreeSASA", "rdkit.Chem.rdFingerprintGenerator", "rdkit.DataStructs",
}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input",
                "globals", "locals", "vars", "getattr", "setattr", "delattr",
                "breakpoint", "memoryview"}
BANNED_ATTRS = {"__globals__", "__code__", "__class__", "__bases__", "__subclasses__",
                "__builtins__", "__loader__", "__spec__"}


def static_check(code: str) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"语法错误: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in {m.split(".")[0] for m in ALLOWED_MODULES}:
                    problems.append(f"禁止 import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in {m.split(".")[0] for m in ALLOWED_MODULES}:
                problems.append(f"禁止 from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"禁止使用 {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(f"禁止访问 {node.attr}")
    if "def compute(" not in code:
        problems.append("必须定义 compute(smiles) 函数")
    return problems


RUNNER = r'''
import json, sys, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Descriptors3D
RDLogger.DisableLog("rdApp.*")
from zhizhi.desc import primitives as prim

USER_CODE

def _main():
    payload = json.loads(open(IN_PATH, encoding="utf-8").read())
    out, errs = {}, {}
    for smi in payload["smiles"]:
        try:
            v = compute(smi)
            if v is None:
                out[smi] = None
            elif isinstance(v, (int, float, np.floating, np.integer)):
                f = float(v)
                out[smi] = None if (math.isnan(f) or math.isinf(f)) else f
            else:
                errs[smi] = f"返回类型 {type(v).__name__}，必须是 float 或 None"
                out[smi] = None
        except Exception as e:
            errs[smi] = f"{type(e).__name__}: {e}"
            out[smi] = None
    json.dump({"values": out, "errors": errs}, open(OUT_PATH, "w", encoding="utf-8"),
              ensure_ascii=False)

_main()
'''


def run_descriptor(code: str, smiles_list: list[str],
                   timeout: int = 900) -> dict:
    """在沙箱里对一批 SMILES 执行 compute()，返回 {values, errors, stats}。"""
    problems = static_check(code)
    if problems:
        return {"ok": False, "static_check": problems,
                "hint": "只能 import numpy/math/rdkit；可直接调用注入的 prim.* 原语库"}

    tmp = Path(tempfile.mkdtemp(prefix="zz_desc_"))
    in_p, out_p, script = tmp / "in.json", tmp / "out.json", tmp / "run.py"
    in_p.write_text(json.dumps({"smiles": smiles_list}, ensure_ascii=False),
                    encoding="utf-8")
    body = (RUNNER
            .replace("PROJECT_ROOT", repr(str(CFG.root)))
            .replace("IN_PATH", repr(str(in_p)))
            .replace("OUT_PATH", repr(str(out_p)))
            .replace("USER_CODE", code))
    script.write_text(body, encoding="utf-8")

    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"执行超时 (>{timeout}s)",
                "hint": "构象生成很贵，考虑减少构象数或简化计算"}
    if not out_p.exists():
        return {"ok": False, "error": "沙箱未产出结果",
                "stderr": (proc.stderr or "")[-2500:], "stdout": (proc.stdout or "")[-800:]}

    res = json.loads(out_p.read_text(encoding="utf-8"))
    vals = res["values"]
    good = [v for v in vals.values() if v is not None]
    import numpy as np
    stats = {"n_total": len(vals), "n_valid": len(good),
             "coverage": round(len(good) / max(len(vals), 1), 3),
             "elapsed_s": round(time.time() - t0, 1)}
    if good:
        arr = np.array(good, float)
        stats.update({"mean": round(float(arr.mean()), 5),
                      "std": round(float(arr.std()), 5),
                      "min": round(float(arr.min()), 5),
                      "max": round(float(arr.max()), 5),
                      "n_unique": int(len(np.unique(np.round(arr, 8))))})
    err_sample = dict(list(res["errors"].items())[:5])
    return {"ok": True, "values": vals, "stats": stats,
            "n_errors": len(res["errors"]), "error_sample": err_sample}
