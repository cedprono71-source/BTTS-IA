
import math

MODELS = [
    "Dixon-Coles",
    "XGBoost",
    "CatBoost",
    "Poisson bivarié",
    "Bayésien"
]


def poisson_probability(lam, goals):
    return (math.exp(-lam) * lam**goals) / math.factorial(goals)


def poisson_btts(home_xg, away_xg):
    return (
        (1 - math.exp(-home_xg)) *
        (1 - math.exp(-away_xg))
    )


def dixon_coles(home_xg, away_xg, rho=-0.05):
    probability = 0

    for h in range(8):
        for a in range(8):
            correction = 1.0

            if h == 0 and a == 0:
                correction = 1 - home_xg * away_xg * rho
            elif h == 0 and a == 1:
                correction = 1 + home_xg * rho
            elif h == 1 and a == 0:
                correction = 1 + away_xg * rho
            elif h == 1 and a == 1:
                correction = 1 - rho

            if h > 0 and a > 0:
                probability += (
                    poisson_probability(home_xg, h) *
                    poisson_probability(away_xg, a) *
                    correction
                )

    return probability


def bivariate_poisson(home_xg, away_xg):
    shared = min(home_xg, away_xg) * 0.10

    home = max(0.01, home_xg - shared)
    away = max(0.01, away_xg - shared)

    return poisson_btts(home, away)


def bayesian_model(home_xg, away_xg):
    raw = poisson_btts(home_xg, away_xg)

    prior = 0.50
    confidence = 0.80

    return raw * confidence + prior * (1 - confidence)


def predict_btts(home_xg, away_xg):
    if not (0.05 <= home_xg <= 5):
        raise ValueError("Home xG invalide")

    if not (0.05 <= away_xg <= 5):
        raise ValueError("Away xG invalide")

    results = {
        "Dixon-Coles": dixon_coles(home_xg, away_xg),
        "XGBoost": None,
        "CatBoost": None,
        "Poisson bivarié": bivariate_poisson(
            home_xg, away_xg
        ),
        "Bayésien": bayesian_model(
            home_xg, away_xg
        )
    }

    available = {
        name: value
        for name, value in results.items()
        if value is not None
    }

    final_probability = sum(
        available.values()
    ) / len(available)

    return {
        "models": {
            name: round(value * 100, 2)
            if value is not None else "En attente d'entraînement"
            for name, value in results.items()
        },
        "btts_yes": round(final_probability * 100, 2),
        "btts_no": round((1-final_probability) * 100, 2),
        "models_available": len(available)
    }

