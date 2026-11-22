
import os
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="BTTS-IA", version="2.0")

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente")
    return psycopg.connect(
        DATABASE_URL,
        connect_timeout=10,
        row_factory=dict_row
    )


def init_db():
    with get_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                home TEXT NOT NULL,
                away TEXT NOT NULL,
                league TEXT NOT NULL,
                date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model TEXT NOT NULL,
                btts_yes DOUBLE PRECISION NOT NULL,
                btts_no DOUBLE PRECISION NOT NULL,
                home_goals INTEGER,
                away_goals INTEGER,
                completed BOOLEAN DEFAULT FALSE
            )
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date
            ON predictions(date)
        """)
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_model
            ON predictions(model)
        """)


class Match(BaseModel):
    home: str
    away: str
    league: str
    btts_yes: float = Field(ge=0, le=100)
    model: str


def prepare_db():
    try:
        init_db()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur Neon : {str(e)}"
        )


@app.get("/")
def home():
    return {
        "status": "BTTS-IA active",
        "version": "2.0",
        "database": "Neon PostgreSQL"
    }


@app.get("/health/db")
def database_health():
    prepare_db()
    try:
        with get_db() as con:
            con.execute("SELECT 1")
        return {"database": "connected", "status": "OK"}
    except Exception as e:
        raise HTTPException(503, detail=str(e))


@app.post("/predict")
def save_prediction(match: Match):
    prepare_db()

    yes = match.btts_yes
    no = 100 - yes

    with get_db() as con:
        cur = con.execute("""
            INSERT INTO predictions
            (home, away, league, model, btts_yes, btts_no)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            match.home,
            match.away,
            match.league,
            match.model,
            yes,
            no
        ))
        prediction_id = cur.fetchone()["id"]

    return {
        "id": prediction_id,
        "btts_yes": yes,
        "btts_no": no,
        "status": "saved"
    }


@app.post("/result/{prediction_id}")
def save_result(
    prediction_id: int,
    home_goals: int = Field(ge=0),
    away_goals: int = Field(ge=0)
):
    prepare_db()

    with get_db() as con:
        cur = con.execute("""
            UPDATE predictions
            SET home_goals = %s,
                away_goals = %s,
                completed = TRUE
            WHERE id = %s
            RETURNING id
        """, (home_goals, away_goals, prediction_id))

        if not cur.fetchone():
            raise HTTPException(404, "Match introuvable")

    return {"status": "result saved"}


@app.get("/stats")
def stats():
    prepare_db()

    with get_db() as con:
        rows = con.execute("""
            SELECT *
            FROM predictions
            WHERE completed = TRUE
        """).fetchall()

    total = len(rows)

    if total == 0:
        return {
            "matches_evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "brier_score": None,
            "calibration": [],
            "by_league": [],
            "by_model": []
        }

    correct = 0
    brier_total = 0
    bins = [[] for _ in range(10)]
    leagues = {}
    models = {}

    for row in rows:
        probability = row["btts_yes"] / 100
        actual = int(
            row["home_goals"] > 0 and
            row["away_goals"] > 0
        )

        predicted = int(probability >= 0.5)
        correct += int(predicted == actual)

        brier_total += (probability - actual) ** 2

        index = min(int(probability * 10), 9)
        bins[index].append((probability, actual))

        for group, key in (
            (leagues, row["league"]),
            (models, row["model"])
        ):
            if key not in group:
                group[key] = {"total": 0, "correct": 0}

            group[key]["total"] += 1
            group[key]["correct"] += int(predicted == actual)

    def grouped_stats(group):
        return [
            {
                "name": name,
                "matches": data["total"],
                "correct": data["correct"],
                "accuracy": round(
                    100 * data["correct"] / data["total"], 2
                ),
                "sample_warning": data["total"] < 30
            }
            for name, data in group.items()
        ]

    calibration = []

    for i, values in enumerate(bins):
        if values:
            calibration.append({
                "probability_range": f"{i*10}-{(i+1)*10}%",
                "matches": len(values),
                "average_probability": round(
                    sum(p for p, _ in values) / len(values) * 100, 2
                ),
                "actual_btts_rate": round(
                    sum(a for _, a in values) / len(values) * 100, 2
                )
            })

    return {
        "matches_evaluated": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2),
        "brier_score": round(brier_total / total, 4),
        "calibration": calibration,
        "by_league": grouped_stats(leagues),
        "by_model": grouped_stats(models)
    }
