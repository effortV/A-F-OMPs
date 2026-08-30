"""致知 ZHIZHI 命令行入口。

  python -m zhizhi.cli doctor            连通性与数据体检
  python -m zhizhi.cli testmodel <id>    换模型前的兼容性体检（6 项）
  python -m zhizhi.cli init              注册核心 59 篇 + 预热模型缓存
  python -m zhizhi.cli ingest            前台跑文献摄取直到队列清空
  python -m zhizhi.cli status            查看进度
  python -m zhizhi.cli chat gewu         与某个智能体交互对话
  python -m zhizhi.cli run gewu "..."    单次提问
  python -m zhizhi.cli report            导出发现报告
  python -m zhizhi.cli ui                启动 Streamlit 网页
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def _p(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_doctor(args) -> None:
    from .core import db, llm
    from .dataio import loader
    db.init()
    print("=== LLM 连通性 ===")
    _p(llm.health_check())
    print("\n=== 数据底座 ===")
    h = loader.data_health()
    _p({k: v for k, v in h.items() if k != "missing_rate_by_feature"})
    print("\n=== 缺失结构（前 8）===")
    _p(dict(list(h["missing_rate_by_feature"].items())[:8]))
    print("\n=== 依赖 ===")
    for m in ("rdkit", "xgboost", "shap", "statsmodels", "pymupdf", "streamlit"):
        try:
            mod = __import__(m)
            print(f"  OK   {m} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  MISS {m} ({e})")


def cmd_testmodel(args) -> None:
    """换模型前先体检：跑一遍系统真正用到的 6 种调用形态。"""
    from .core import db
    from .core.modelcheck import check_current, check_model
    db.init()
    res = check_model(args.model) if args.model else check_current()
    print(f"模型：{res['model']}")
    print(f"端点：{res['base_url']}\n")
    for name, c in res["checks"].items():
        flag = "OK  " if c["ok"] else "FAIL"
        extra = ", ".join(f"{k}={v}" for k, v in c.items()
                          if k not in ("ok", "seconds", "error"))
        print(f"  [{flag}] {name:12s} {c['seconds']:5.1f}s  "
              f"{extra or c.get('error', '')}")
    print(f"\n判定：{res['verdict']}")
    for a in res["advice"]:
        print(f"  · {a}")


def cmd_init(args) -> None:
    from .core import db
    from .lit import worker
    from .ml import model as M
    db.init()
    print("注册核心语料…")
    _p(worker.bootstrap_core_corpus())
    _p(worker.scan_new_pdfs())
    print("\n预热模型（分组 CV，首次约 1-2 分钟）…")
    _p(M.legacy_report())
    print("\n完成。下一步：python -m zhizhi.cli ingest   或   python -m zhizhi.cli ui")


def cmd_ingest(args) -> None:
    from .core import db
    from .lit import worker
    db.init()
    worker.bootstrap_core_corpus()
    if args.workers:
        from .core.config import CFG
        CFG["literature"]["n_workers"] = args.workers
    worker.control("start")
    print(f"摄取已启动（{args.workers or CFG_workers()} 线程）。Ctrl+C 可随时中断，进度已落库。")
    done_before = -1
    try:
        while True:
            st = worker.status()
            q = st["queue"]
            left = q.get("queued", 0) + q.get("running", 0)
            done = q.get("done", 0)
            if done != done_before:
                cur = ", ".join(f"{r['label'][:40]}" for r in st.get("running_now", []))
                print(f"  核心语料 {st['core_corpus_progress']} | 完成 {done} | 剩余 {left}"
                      f" | 进行中: {cur}")
                done_before = done
            if left == 0:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        worker.control("pause")
        print("\n已暂停。下次运行会从断点继续。")
        return
    print("\n队列清空。")
    _p(worker.status())


def CFG_workers() -> int:
    from .core.config import CFG
    return int(CFG.get("literature.n_workers", 4))


def cmd_status(args) -> None:
    from .core import db
    from .tools import meta_tools
    db.init()
    _p(meta_tools.system_overview())


def cmd_run(args) -> None:
    from .core import db
    from .core.agent import new_session
    from .agents.registry import get_agent
    db.init()
    a = get_agent(args.agent)
    if a is None:
        print(f"未知智能体 {args.agent}；可选 bowen / liangheng / gewu / yanzhen")
        return
    sid = args.session or new_session(args.agent)
    for ev in a.run(sid, args.prompt):
        if ev["type"] == "tool_call":
            print(f"\n  ⚙ {ev['name']}({json.dumps(ev['args'], ensure_ascii=False)[:160]})",
                  flush=True)
        elif ev["type"] == "tool_result":
            print(f"  ↳ {ev['text'][:400]}", flush=True)
        elif ev["type"] == "text":
            print(f"\n{ev['text']}\n", flush=True)
        elif ev["type"] == "error":
            print(f"\n[错误] {ev['text']}\n", flush=True)
    print(f"(session: {sid})")


def cmd_chat(args) -> None:
    from .core import db
    from .core.agent import new_session
    from .agents.registry import META, get_agent
    db.init()
    a = get_agent(args.agent)
    if a is None:
        print(f"未知智能体 {args.agent}")
        return
    m = META[args.agent]
    sid = args.session or new_session(args.agent)
    print(f"{m['icon']} {m['cn']} {m['en']} · {m['role']}\n{m['desc']}")
    print(f"session={sid}　输入 exit 退出，输入 /tools 看工具清单\n")
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("exit", "quit", "q"):
            break
        if q == "/tools":
            from .core.tools import REGISTRY
            print(REGISTRY.describe(a.tool_names))
            continue
        for ev in a.run(sid, q):
            if ev["type"] == "tool_call":
                print(f"  ⚙ {ev['name']} "
                      f"{json.dumps(ev['args'], ensure_ascii=False)[:140]}", flush=True)
            elif ev["type"] == "text":
                print(f"\n{m['cn']} > {ev['text']}\n", flush=True)
            elif ev["type"] == "error":
                print(f"\n[错误] {ev['text']}\n", flush=True)


def cmd_report(args) -> None:
    from .core import db
    from .tools import meta_tools
    db.init()
    _p(meta_tools.export_report())


def cmd_ui(args) -> None:
    import subprocess
    from pathlib import Path
    app = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                    "--server.headless", "true"], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(prog="zhizhi", description="致知 ZHIZHI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("ui").set_defaults(fn=cmd_ui)

    tm = sub.add_parser("testmodel", help="换模型前的兼容性体检")
    tm.add_argument("model", nargs="?", default="",
                    help="要测的模型 id，留空则测当前 .env 里的")
    tm.set_defaults(fn=cmd_testmodel)

    g = sub.add_parser("ingest")
    g.add_argument("--workers", type=int, default=0)
    g.set_defaults(fn=cmd_ingest)

    r = sub.add_parser("run")
    r.add_argument("agent")
    r.add_argument("prompt")
    r.add_argument("--session", default="")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("chat")
    c.add_argument("agent")
    c.add_argument("--session", default="")
    c.set_defaults(fn=cmd_chat)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
