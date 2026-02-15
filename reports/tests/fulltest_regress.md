# Socratic Git Regression Report

## Question
When was SparkSession introduced in ex01_data_retrieval/src/main/scala/Main.scala?

## Introducing Commit
- hash: 415603667981e143586bfe8700c58ae926b7ee26
- author: Goulfrom
- date: 2026-01-03
- message: test

## Evidence Snippets
- changed_files: ex01_data_retrieval/src/main/scala/Main.scala
- matched snippet (+/- 5 lines):
```text
   1: import org.apache.spark.sql.SparkSession
   2: 
   3: object Main {
   4:   def main(args: Array[String]): Unit = {
   5: 
   6:     // 默认用 2025-01，可以通过命令行参数覆盖
```
- diff_snippet:
```diff
diff --git a/ex01_data_retrieval/src/main/scala/Main.scala b/ex01_data_retrieval/src/main/scala/Main.scala
new file mode 100644
index 0000000..f134891
--- /dev/null
+++ b/ex01_data_retrieval/src/main/scala/Main.scala
@@ -0,0 +1,36 @@
+import org.apache.spark.sql.SparkSession
+
+object Main {
+  def main(args: Array[String]): Unit = {
+
+    // 默认用 2025-01，可以通过命令行参数覆盖
+    val year  = if (args.length > 0) args(0) else "2025"
+    val month = if (args.length > 1) args(1) else "01"
+
+    val fileName  = s"yellow_tripdata_${year}-${month}.parquet"
+    val localPath = s"data/raw/$fileName"
+    val s3Path    = s"s3a://nyc-raw/$fileName"
+
+    val spark = SparkSession.builder()
+      .appName("Ex01DataRetrieval")
+      .master("local[*]")
+      .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
+      .config("spark.hadoop.fs.s3a.access.key", "minio")
+      .config("spark.hadoop.fs.s3a.secret.key", "minio123")
+      .config("spark.hadoop.fs.s3a.path.style.access", "true")
+      .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
+      .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
+      .getOrCreate()
+
+    val df = spark.read.parquet(localPath)
+
+    df.printSchema()
+    df.show(5)
+
+    df.write
+      .mode("overwrite")
+      .parquet(s3Path)
+
+    spark.stop()
```

## Local History Window
- 69c01059f2f50a0375d8a67b83a98c50ba32b423 2026-02-01 Ajout d'automatisation et utilisation par défaut de Java 11.
- 47e8e27c41c81511ba07c362e6c9cc20d1dfd5b0 2026-01-31 EX3 schéma + tables du Data Warehouse est fini
- 415603667981e143586bfe8700c58ae926b7ee26 2026-01-03 test

## Answer (Evidence-Driven)
Findings (Deterministic):
SparkSession was introduced in commit 415603667981e143586bfe8700c58ae926b7ee26 on 2026-01-03 by Goulfrom in ex01_data_retrieval/src/main/scala/Main.scala. Key snippet: import org.apache.spark.sql.SparkSession

Motive (Evidence-Driven, Optional):
Motive is not explicitly stated in commit message/diff.

## Limitations
- Commit message/diff do not explicitly explain motive.
