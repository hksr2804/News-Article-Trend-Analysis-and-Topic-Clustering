"""
src/ingest.py — PySpark batch ingestion

Loads the raw JSON dataset, filters incomplete rows, combines headline +
short_description into a single text field, casts the date column to DateType,
and saves the result to output/clean_data.parquet (partitioned Parquet folder).
"""

import sys
import os

# Allow running from repo root: python -m src.ingest or python src/ingest.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows: PySpark requires winutils.exe + hadoop.dll in HADOOP_HOME/bin.
# Also add that bin dir to PATH so the JVM can load hadoop.dll as a native lib.
if os.name == "nt":
    if not os.environ.get("HADOOP_HOME"):
        os.environ["HADOOP_HOME"] = r"C:\hadoop"
    hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import config


def run():
    spark = (
        SparkSession.builder
        .appName("NewsIngest")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"[ingest] Reading {config.DATA_PATH} ...")
        df = spark.read.json(config.DATA_PATH)

        total = df.count()
        print(f"[ingest] Total rows loaded: {total:,}")

        # Keep rows where headline, short_description, and date are all non-null/non-empty
        df_clean = df.filter(
            F.col("headline").isNotNull() & (F.trim(F.col("headline")) != "") &
            F.col("short_description").isNotNull() & (F.trim(F.col("short_description")) != "") &
            F.col("date").isNotNull() & (F.trim(F.col("date")) != "")
        )

        kept = df_clean.count()
        print(f"[ingest] Rows after filtering: {kept:,}  (dropped {total - kept:,})")

        # Combine headline + short_description into a single text field
        df_clean = df_clean.withColumn(
            "text",
            F.concat(F.col("headline"), F.lit(" "), F.col("short_description"))
        )

        # Cast date string to DateType — malformed dates silently become null
        df_clean = df_clean.withColumn("date", F.col("date").cast("date"))

        unparseable = df_clean.filter(F.col("date").isNull()).count()
        if unparseable > 0:
            print(f"[ingest] WARNING: {unparseable:,} rows have an unparseable date and will be dropped.")
            df_clean = df_clean.filter(F.col("date").isNotNull())

        # Keep only the two columns downstream steps need
        df_out = df_clean.select("text", "date")

        print(f"[ingest] Saving to {config.CLEAN_DATA_PATH} ...")
        df_out.write.mode("overwrite").parquet(config.CLEAN_DATA_PATH)

        print("[ingest] Done.")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
