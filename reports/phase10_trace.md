# Socratic Git Trace Report

## Question
In src/main.py, why was hybrid_search added?

## Detected Target
- file: src/main.py
- symbol: hybrid_search
- line: 328
- resolved_line: 328

## Introducing Commit
- hash: eb01f1b12d1ca6d77dc3635d7257ccb596c6247a
- author: SihamDaanouni
- date: 2026-01-23
- message: updating src files

## Introducing Commit Details
- hash: eb01f1b12d1ca6d77dc3635d7257ccb596c6247a
- author/date: SihamDaanouni / 2026-01-23
- message (first 10 lines):
```text
updating src files
```
- changed_files: src/build_index.py, src/docs_to_corpus.py, src/evaluate.py, src/main.py, src/search_index_test.py
- diff_snippet (clean_diff + first 120 lines):
```diff
diff --git a/src/build_index.py b/src/build_index.py
index f158bed..2ba90e6 100644
--- a/src/build_index.py
+++ b/src/build_index.py
@@ -1,4 +1,5 @@
 from pathlib import Path
+import sys
 
 import faiss
 import numpy as np
@@ -6,9 +7,18 @@ import pandas as pd
 from sentence_transformers import SentenceTransformer
 
 # Project root directory
-BASE_DIR = Path(__file__).resolve().parent.parent
+sys.path.append("..") 
+
+# CONFIGURATION DES CHEMINS
+BASE_DIR = Path("..")
+
+# Dossier où se trouve le CSV (Données traitées)
 PROC_DIR = BASE_DIR / "data" / "processed"
+
+# Dossier où on va sauvegarder l'index
 INDEX_DIR = BASE_DIR / "data" / "index"
+
+# Création du dossier s'il n'existe pas
 INDEX_DIR.mkdir(parents=True, exist_ok=True)
 
 
@@ -21,12 +31,12 @@ def main():
     texts = df["text"].astype(str).tolist()
     doc_ids = df["doc_id"].tolist()
 
-    # 1) Load sentence embedding model
+    # 1. Chargement du modèle de sentence embedding
     model_name = "sentence-transformers/all-MiniLM-L6-v2"
     print(f"Loaded embedding model: {model_name}")
     model = SentenceTransformer(model_name)
 
diff --git a/src/docs_to_corpus.py b/src/docs_to_corpus.py
index aa2680c..fcd1d05 100644
--- a/src/docs_to_corpus.py
+++ b/src/docs_to_corpus.py
@@ -1,33 +1,46 @@
+# IMPORTS
 from pathlib import Path
+import sys
 import re
 import json
 import pandas as pd
+import csv
 
-# CONFIGURATION DES CHEMINS (ETL SETUP)
+import faiss
+import numpy as np
+from sentence_transformers import SentenceTransformer
 
-# On remonte d'un niveau (../) pour atteindre la racine du projet
-BASE_DIR = Path(__file__).resolve().parent.parent 
 
-# Dossier d'entrée (Documents bruts)
-# Assure-toi que tes PDFs/TXT sont bien ici
-RAW_DOC_DIR = BASE_DIR / "data" / "raw"
 
-# Dossier de sortie (Données traitées)
+
+from pathlib import Path
+
+sys.path.append("..") 
+
+# CONFIGURATION DES CHEMINS
+BASE_DIR = Path("..") 
+
+# Dossier d'entrée
+RAW_DOC_DIR = BASE_DIR / "data" / "raw_docs"
+
+# Dossier de sortie
 PROC_DIR = BASE_DIR / "data" / "processed"
+
diff --git a/src/evaluate.py b/src/evaluate.py
index 113fbb7..e91626d 100644
--- a/src/evaluate.py
+++ b/src/evaluate.py
@@ -1,181 +1,282 @@
+import sys
 import json
 import pandas as pd
 import numpy as np
 from pathlib import Path
 from tqdm import tqdm
-import sys
+import faiss
+from sentence_transformers import SentenceTransformer, CrossEncoder
+from rank_bm25 import BM25Okapi
 
-# CONFIGURATION ET IMPORTS
-
-# On a besoin d'importer les briques technologiques depuis main.py
-# (FAISS, BM25, Fusion, Reranking)
-try:
-    sys.path.append("src")
-    from main import (
-        load_resources, 
-        retrieve_faiss, 
-        retrieve_bm25, 
-        reciprocal_rank_fusion, 
-        rerank_contexts
-    )
-except ImportError:
-    # Fallback si lancé depuis le dossier src directement
-    from main import (
-        load_resources, 
-        retrieve_faiss, 
-        retrieve_bm25, 
-        reciprocal_rank_fusion, 
-        rerank_contexts
-    )
-
-# Chemins des fichiers
```

## Local History Window
- eb01f1b12d1ca6d77dc3635d7257ccb596c6247a 2026-01-23 updating src files
- b3717eb81e805fcbf146dd3bc610c461b4b4c343 2025-12-22 git update commentaries and adding hybrid search (FAISS and BM25)
- d017e6004f6f66135b789bb547b683c30550fe95 2025-12-22 Adding dynamic threshold in main.py and evaluate.py
- 4a12803d16b908fafcd83ba3ddfa150a0af29c20 2025-12-11 Adding multi-query and evaluation
- 4ec29fab24fea3e3271a5cb8d1d21fce6944602d 2025-12-05 Add document-based RAG pipeline and baseline fallback，Update RAG demo with docs corpus and similarity threshold，Refine main RAG flow and add baseline LLM mode

## Retrieved Commits (TopK)
- [1] b3717eb81e805fcbf146dd3bc610c461b4b4c343 2025-12-22 Amine
  - message: git update commentaries and adding hybrid search (FAISS and BM25)
  - files: (none)
- [2] 5cf7c26668da566c0f106d33a016094de24537fb 2025-12-05 Goulfrom
  - message: Rename rag_qa.py to main.py in README
  - files: (none)
- [3] d017e6004f6f66135b789bb547b683c30550fe95 2025-12-22 Amine
  - message: Adding dynamic threshold in main.py and evaluate.py
  - files: (none)

## Evidence Summary
- eb01f1b12d1ca6d77dc3635d7257ccb596c6247a
- b3717eb81e805fcbf146dd3bc610c461b4b4c343
- d017e6004f6f66135b789bb547b683c30550fe95
- 4a12803d16b908fafcd83ba3ddfa150a0af29c20
- 4ec29fab24fea3e3271a5cb8d1d21fce6944602d
- 5cf7c26668da566c0f106d33a016094de24537fb

## Answer (Evidence-Driven)
Evidence Hashes: eb01f1b12d1ca6d77dc3635d7257ccb596c6247a, b3717eb81e805fcbf146dd3bc610c461b4b4c343, d017e6004f6f66135b789bb547b683c30550fe95, 4a12803d16b908fafcd83ba3ddfa150a0af29c20, 4ec29fab24fea3e3271a5cb8d1d21fce6944602d, 5cf7c26668da566c0f106d33a016094de24537fb Evidence Hashes: none Conclusion: I don't know (insufficient evidence in commit message/diff). ```<|im_end|>
<|im_start|>

## Limitations
- No major limitations detected for this trace run.
