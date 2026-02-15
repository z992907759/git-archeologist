# Socratic Git Bisect Report

## Inputs
- repo: /tmp/socratic_full_test
- good: 353af9dd88aad637d724733a8811d6c264bc3073
- bad: 05c3ed9846b48998479c15b57b1524ce6d2d9428
- cmd: `python -c "import subprocess,sys; h=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); sys.exit(1 if h=='05c3ed9846b48998479c15b57b1524ce6d2d9428' else 0)"`
- max_steps: 30
- safety_note: `--cmd` executes arbitrary shell commands; use at your own risk.

## Reproducibility
- good_returncode: 0
- bad_returncode: 1

## Steps
| commit | pass_fail | returncode | note |
|---|---|---|---|
| 353af9dd88aad637d724733a8811d6c264bc3073 | pass | 0 | boundary=good |
| 05c3ed9846b48998479c15b57b1524ce6d2d9428 | fail | 1 | boundary=bad |
| 05c3ed9846b48998479c15b57b1524ce6d2d9428 | fail | 1 | mid |

## First Bad Commit
- hash: 05c3ed9846b48998479c15b57b1524ce6d2d9428
- author: Goulfrom
- date: 2026-02-13
- message: docs: clarify EX6 startup order and postgres readiness checks
- changed_files: README.md
- diff_snippet:
```diff
diff --git a/README.md b/README.md
index 8008553..4039769 100644
--- a/README.md
+++ b/README.md
@@ -96,17 +96,40 @@ Les fichiers Airflow de l'exercice sont dans `ex06_airflow/airflow/` (DAGs, logs
 
 Pre-requis important:
 ```bash
-# demarrer d'abord la stack de base (cree le reseau spark-network + postgres-db)
-docker compose up -d
+# demarrer explicitement les services de base requis par EX6
+docker compose up -d postgres spark-master spark-worker-1 spark-worker-2 minio
+
+```
+
+Initialisation des permissions (obligatoire sur Linux, recommande partout):
+```bash
+echo "AIRFLOW_UID=$(id -u)" > .env
+mkdir -p ex06_airflow/airflow/dags ex06_airflow/airflow/logs ex06_airflow/airflow/plugins
+sudo chown -R $(id -u):0 ex06_airflow/airflow/dags ex06_airflow/airflow/logs ex06_airflow/airflow/plugins
+sudo chmod -R 775 ex06_airflow/airflow/dags ex06_airflow/airflow/logs ex06_airflow/airflow/plugins
 ```
 
 Depuis la racine du repo:
 ```bash
 export PROJECT_DIR="$(pwd)"
 export AIRFLOW_DOCKER_NETWORK="bigdata_projet_spark-network"
+docker compose -f docker-compose.airflow.yml up airflow-init
 docker compose -f docker-compose.airflow.yml up -d
 ```
 
+Verification rapide apres demarrage:
+```bash
+docker compose -f docker-compose.airflow.yml ps
+docker ps --filter name=postgres-db
+```
+
+Si `airflow-init` termine avec `exit 1`:
+```bash
+docker compose -f docker-compose.airflow.yml logs airflow-init
```

## Failure Log Excerpt
### stdout_tail
```text
(empty)
```
### stderr_tail
```text
(empty)
```

## Findings
First bad commit is 05c3ed9846b48998479c15b57b1524ce6d2d9428 (date=2026-02-13, author=Goulfrom) under command: python -c "import subprocess,sys; h=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); sys.exit(1 if h=='05c3ed9846b48998479c15b57b1524ce6d2d9428' else 0)" Only 1 candidate commit in range; no mid-point tests were necessary.

## Motive
Motive is not explicitly stated in commit message/diff.
