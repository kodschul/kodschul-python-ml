"""
Big Data Example: Word Count with PySpark

This script outlines how to perform a simple word count using PySpark's RDD API. It requires a
running Spark environment. If executed in a Spark environment, it reads a text file, splits
lines into words, counts occurrences, and prints the top results.

Dependencies:
    - pyspark

Usage:
    spark-submit big_data_pyspark_example.py <input_text_file>

Note: This script is provided for educational purposes and assumes PySpark is installed and
configured. It will not run in a non-Spark environment.
"""
import sys
from pyspark import SparkContext

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: spark-submit big_data_pyspark_example.py <input_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    sc = SparkContext(appName="WordCountExample")
    text_rdd = sc.textFile(input_path)
    counts = (
        text_rdd.flatMap(lambda line: line.split())
        .map(lambda word: (word.lower(), 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda pair: -pair[1])
    )
    for word, count in counts.take(20):
        print(f"{word}: {count}")
    sc.stop()
