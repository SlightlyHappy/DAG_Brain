# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from .schemas import ProjectState, now_iso
from .seed_data import get_seed_projects
from .config import DATA_DIR


DB_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "dag_brain.sqlite")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                discipline TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL
            )
            """
        )
        row = conn.execute("SELECT COUNT(*) AS cnt FROM projects").fetchone()
        if row and row["cnt"] == 0:
            for seed_proj in get_seed_projects():
                save_project(seed_proj, conn=conn)


def list_projects() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, discipline, updated_at, state_json FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        summaries = []
        for r in rows:
            data = json.loads(r["state_json"])
            summaries.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "discipline": r["discipline"],
                    "updated_at": r["updated_at"],
                    "node_count": len(data.get("nodes", [])),
                    "edge_count": len(data.get("edges", [])),
                    "source_count": len(data.get("sources", [])),
                    "citation_style": data.get("citation_style", "apa"),
                }
            )
        return summaries


def get_project(project_id: str) -> Optional[ProjectState]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT state_json FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if not row:
            return None
        return ProjectState.model_validate_json(row["state_json"])


def save_project(
    project: ProjectState, conn: Optional[sqlite3.Connection] = None
) -> ProjectState:
    project.updated_at = now_iso()
    payload = project.model_dump_json()
    if conn is not None:
        conn.execute(
            """
            INSERT INTO projects (id, title, discipline, updated_at, state_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                discipline = excluded.discipline,
                updated_at = excluded.updated_at,
                state_json = excluded.state_json
            """,
            (project.id, project.title, project.discipline, project.updated_at, payload),
        )
    else:
        with get_conn() as c:
            c.execute(
                """
                INSERT INTO projects (id, title, discipline, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    discipline = excluded.discipline,
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
                """,
                (project.id, project.title, project.discipline, project.updated_at, payload),
            )
    return project


def delete_project(project_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cur.rowcount > 0
