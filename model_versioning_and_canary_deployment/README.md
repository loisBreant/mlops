# TP2 — MLflow Model Service (compact)

## Env vars
- MODEL_URI (requis) — URI du modèle à charger au démarrage
  - Ex: `models:/m-0600ab11415a4868848f2f81cefb219a` ou `runs:/<run_id>/model`
  - Ne pas ajouter de guillemets dans docker-compose
- MLFLOW_TRACKING_URI (recommandé) — doit pointer vers le dossier MLflow monté dans le container
  - Ex: `file:///home/lois/Documents/mlops/tp2/mlruns`
- BASE_URL (optionnel, tests) — défaut: `http://localhost:8000`
- MODEL_URI_TO_LOAD (optionnel, tests) — URI pour `/update-model`

## docker-compose (extrait)
```yaml
services:
  model-service:
    environment:
      MLFLOW_TRACKING_URI: file:///home/lois/Documents/mlops/tp2/mlruns
      MODEL_URI: models:/m-<model-id>
    volumes:
      - /home/lois/Documents/mlops/tp2/mlruns:/home/lois/Documents/mlops/tp2/mlruns
```

## Tests locaux (zsh)
```zsh
export BASE_URL=http://localhost:8000
export MODEL_URI_TO_LOAD=models:/m-<model-id>
python tp2/test_predict.py
python tp2/test_update.py
```

## Démarrage
```zsh
docker compose up --build
```

## Entraînement (optionnel)
```zsh
python tp2/train_iris_mlflow.py --save_model
```
