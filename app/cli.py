from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import func, select

from app.db import SessionLocal, init_db
from app.models import Conversation, Document, KnowledgeBase, TrainingTrace


def export_training(output: Path, approved_only: bool) -> None:
    init_db()
    output.parent.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    try:
        statement = select(TrainingTrace).order_by(TrainingTrace.created_at)
        if approved_only:
            statement = statement.where(TrainingTrace.approved_for_training.is_(True))
        with output.open("w", encoding="utf-8") as handle:
            for row in db.scalars(statement):
                payload = {
                    "instruction": row.instruction,
                    "input": row.input_text,
                    "output": row.human_revision or row.output_text,
                    "metadata": {
                        "conversation_id": row.conversation_id,
                        "rating": row.rating,
                        "approved_for_training": row.approved_for_training,
                        "retrieval_evidence": row.retrieval_evidence,
                        "tool_trace": row.tool_trace,
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        db.close()
    print(f"Training JSONL exported to {output.resolve()}")


def show_status() -> None:
    init_db()
    db = SessionLocal()
    try:
        counts = {
            "knowledge_bases": db.scalar(select(func.count()).select_from(KnowledgeBase)),
            "conversations": db.scalar(select(func.count()).select_from(Conversation)),
            "documents": db.scalar(select(func.count()).select_from(Document)),
            "approved_training_traces": db.scalar(
                select(func.count()).select_from(TrainingTrace).where(TrainingTrace.approved_for_training.is_(True))
            ),
        }
        print(json.dumps(counts, ensure_ascii=False, indent=2))
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="nf-atlas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="初始化数据库表")
    subparsers.add_parser("status", help="显示知识库计数")
    export = subparsers.add_parser("export-training", help="导出 instruction/input/output JSONL")
    export.add_argument("--output", type=Path, default=Path("data/runtime/exports/training.jsonl"))
    export.add_argument("--include-unapproved", action="store_true")
    args = parser.parse_args()
    if args.command == "init-db":
        init_db()
        print("Database initialized")
    elif args.command == "status":
        show_status()
    elif args.command == "export-training":
        export_training(args.output, not args.include_unapproved)


if __name__ == "__main__":
    main()
