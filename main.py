
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime

app = FastAPI(title="BTTS-IA", version="1.0")

DB = "btts.db"

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home TEXT,
            away TEXT,
            league TEXT,
            date TEXT,
            model TEXT,
            btts_yes REAL,
            btts_no REAL,
            home_goals INTEGER,
            away_goals INTEGER,
            completed INTEGER DEFAULT 0
        )
        """)

init_db()

class Match(BaseModel):
    home: str
    away: str
    league: str
    btts_yes: float
    model: str

@app.get("/")
def home():
    return {"status": "BTTS-IA active", "version": "1.0"}

@app.post("/predict")
def save_prediction(match: Match):
    yes = max(0, min(100, match.btts_yes))
    with sqlite3.connect(DB) as con:
        cur = con.execute("""
        INSERT INTO predictions
        (home, away, league, date, model, btts_yes, btts_no)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            match.home, match.away, match.league,
            datetime.now().isoformat(),
            match.model, yes, 100 - yes
        ))
        return {"id": cur.lastrowid, "btts_yes": yes,
                "btts_no": 100 - yes}

@app.post("/result/{prediction_id}")
def save_result(prediction_id: int,
                home_goals: int, away_goals: int):
    with sqlite3.connect(DB) as con:
        cur = con.execute("""
        UPDATE predictions
        SET home_goals=?, away_goals=?, completed=1
        WHERE id=?
        """, (home_goals, away_goals, prediction_id))
        if cur.rowcount == 0:
            return {"error": "Match introuvable"}
    return {"status": "Résultat enregistré"}

@app.get("/stats")
def stats():
    with sqlite3.connect(DB) as con:
        total = con.execute("""
        SELECT COUNT(*) FROM predictions WHERE completed=1
        """).fetchone()[0]

        correct = con.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE completed=1 AND
        ((btts_yes >= 50 AND home_goals > 0 AND away_goals > 0)
        OR
        (btts_yes < 50 AND (home_goals = 0 OR away_goals = 0)))
        """).fetchone()[0]

    return {
        "matches_evaluated": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2)
        if total else None
    }

