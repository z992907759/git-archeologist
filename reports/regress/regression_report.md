# Socratic Git Regression Report

## Question
When was HYBRID introduced in src/main.py?

## Introducing Commit
- hash: b3717eb81e805fcbf146dd3bc610c461b4b4c343
- author: Amine
- date: 2025-12-22
- message: git update commentaries and adding hybrid search (FAISS and BM25)

## Evidence Snippets
- changed_files: src/docs_to_corpus.py, src/evaluate.py, src/main.py, src/plain_raw_qa.py
- matched snippet (+/- 5 lines):
```text
 280:     if not final_docs or final_docs[0]['score'] < THRESHOLD_RERANK:
 281:         return _fallback(query, "Low Rerank Score (< 0.0)")
 282:         
 283:     return call_llm(build_prompt(query, final_docs))
 284: 
 285: # V4: HYBRID ULTIMATE (La Totale)
 286: def rag_v4_hybrid(query, df, index, embed_model, bm25, reranker, top_k=5):
 287:     print("=== [V4] Pipeline: Multi-Query -> Hybrid (FAISS+BM25) -> RRF -> Reranking ===")
 288:     
 289:     # 1. Expansion
 290:     variations = [query]
```
- diff_snippet:
```diff
diff --git a/src/docs_to_corpus.py b/src/docs_to_corpus.py
index 54824d2..aa2680c 100644
--- a/src/docs_to_corpus.py
+++ b/src/docs_to_corpus.py
@@ -3,17 +3,31 @@ import re
 import json
 import pandas as pd
 
-PROC_DIR = Path("../data/processed")
+# CONFIGURATION DES CHEMINS (ETL SETUP)
+
+# On remonte d'un niveau (../) pour atteindre la racine du projet
+BASE_DIR = Path(__file__).resolve().parent.parent 
+
+# Dossier d'entrée (Documents bruts)
+# Assure-toi que tes PDFs/TXT sont bien ici
+RAW_DOC_DIR = BASE_DIR / "data" / "raw"
+
+# Dossier de sortie (Données traitées)
+PROC_DIR = BASE_DIR / "data" / "processed"
 PROC_DIR.mkdir(parents=True, exist_ok=True)
 
+# Fichier final (Corpus structuré)
 OUT_PATH = PROC_DIR / "docs_corpus.csv"
 
 
+# 1. FONCTIONS D'EXTRACTION (READERS)
+
 def read_txt(path: Path) -> str:
     return path.read_text(encoding="utf-8", errors="ignore")
 
 
 def read_pdf(path: Path) -> str:
+
     try:
         from PyPDF2 import PdfReader
     except ImportError:
@@ -24,6 +38,7 @@ def read_pdf(path: Path) -> str:
     try:
         reader = PdfReader(str(path))
diff --git a/src/evaluate.py b/src/evaluate.py
index b3e9f7b..113fbb7 100644
--- a/src/evaluate.py
+++ b/src/evaluate.py
@@ -5,102 +5,141 @@ from pathlib import Path
 from tqdm import tqdm
 import sys
 
+# CONFIGURATION ET IMPORTS
 
-# Mettre à True pour tester le Multi-Query, False pour la recherche simple
-SIMILARITY_THRESHOLD = 0.5
-USE_MULTI_QUERY = True 
-OUTPUT_FILENAME = "evaluation_results_multiquery.csv" if USE_MULTI_QUERY else "evaluation_results_baseline.csv"
-
-
-# On essaie d'importer les fonctions depuis main.py
+# On a besoin d'importer les briques technologiques depuis main.py
+# (FAISS, BM25, Fusion, Reranking)
 try:
-    from main import load_resources, retrieve, retrieve_multi_query
-except ImportError:
     sys.path.append("src")
-    from main import load_resources, retrieve, retrieve_multi_query
+    from main import (
+        load_resources, 
+        retrieve_faiss, 
+        retrieve_bm25, 
+        reciprocal_rank_fusion, 
+        rerank_contexts
+    )
+except ImportError:
+    # Fallback si lancé depuis le dossier src directement
+    from main import (
+        load_resources, 
+        retrieve_faiss, 
+        retrieve_bm25, 
+        reciprocal_rank_fusion, 
+        rerank_contexts
+    )
diff --git a/src/main.py b/src/main.py
index 2ba94d5..0ba176e 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,424 +1,360 @@
 from pathlib import Path
 import textwrap
+import warnings
 
+# Bibliothèques de calcul et data
 import faiss
 import numpy as np
 import pandas as pd
-from sentence_transformers import SentenceTransformer
-from transformers import AutoTokenizer, AutoModelForCausalLM
 import torch
-import warnings
 
-# ====== CONFIGURATION GLOBALE ======
+# Bibliothèques NLP (Sémantique & Reranking)
+from sentence_transformers import SentenceTransformer, CrossEncoder
+from transformers import AutoTokenizer, AutoModelForCausalLM
+
+# Bibliothèque NLP (Mots-clés)
+from rank_bm25 import BM25Okapi 
+
+# =========================================================
+# CONFIGURATION GLOBALE
+# =========================================================
+
+# Le modèle LLM (Cerveau) : Version légère (1B) mais instruction-tuned
 MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
-DEFAULT_THRESHOLD = 0.50  # Seuil de pertinence initial
 
-# ===================================
+# Seuils de décision (Gardes-fous anti-hallucination)
+THRESHOLD_SIMPLE = 0.45   # Pour FAISS (0 à 1). En dessous, c'est du bruit.
+THRESHOLD_RERANK = 0.00   # Pour Cross-Encoder (Logits -10 à +10). < 0 signifie "Non pertinent".
 
-print(f"Loading LLM model: {MODEL_NAME}")
```

## Local History Window
- eb01f1b12d1ca6d77dc3635d7257ccb596c6247a 2026-01-23 updating src files
- b3717eb81e805fcbf146dd3bc610c461b4b4c343 2025-12-22 git update commentaries and adding hybrid search (FAISS and BM25)
- d017e6004f6f66135b789bb547b683c30550fe95 2025-12-22 Adding dynamic threshold in main.py and evaluate.py
- 4a12803d16b908fafcd83ba3ddfa150a0af29c20 2025-12-11 Adding multi-query and evaluation

## Answer (Evidence-Driven)
Findings (Deterministic):
HYBRID was introduced in commit b3717eb81e805fcbf146dd3bc610c461b4b4c343 on 2025-12-22 by Amine in src/main.py. Key snippet: if not final_docs or final_docs[0]['score'] < THRESHOLD_RERANK:

Motive (Evidence-Driven, Optional):
Motive is not explicitly stated in commit message/diff.

## Limitations
- Commit message/diff do not explicitly explain motive.
