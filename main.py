"""ДЗ 3: шість етапів аналізу CSV за допомогою PySpark DataFrame API."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DateType, DecimalType, IntegerType, StringType, StructField, StructType,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"


def show_result(df, filename, title, rows=20):
    """Вивести справжній df.show() та зберегти той самий текст для скриншотів."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        print(title)
        print(f"Rows: {df.count()}")
        df.show(rows, truncate=False)
    text = buffer.getvalue()
    print(text, end="")
    (OUTPUT / f"{filename}.txt").write_text(text, encoding="utf-8")


def main():
    OUTPUT.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("goit-de-hw-03")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        # 1. Кожен CSV читається як окремий Spark DataFrame.
        # DecimalType забезпечує точне представлення грошових сум.
        users_schema = StructType([
            StructField("user_id", IntegerType()),
            StructField("name", StringType()),
            StructField("age", IntegerType()),
            StructField("email", StringType()),
        ])
        purchases_schema = StructType([
            StructField("purchase_id", IntegerType()),
            StructField("user_id", IntegerType()),
            StructField("product_id", IntegerType()),
            StructField("date", DateType()),
            StructField("quantity", IntegerType()),
        ])
        products_schema = StructType([
            StructField("product_id", IntegerType()),
            StructField("product_name", StringType()),
            StructField("category", StringType()),
            StructField("price", DecimalType(12, 2)),
        ])

        def read_csv(name, schema):
            return (
                spark.read.option("header", True)
                .option("dateFormat", "yyyy-MM-dd")
                .option("ignoreLeadingWhiteSpace", True)
                .option("ignoreTrailingWhiteSpace", True)
                .option("nullValue", "")
                .schema(schema).csv(str(ROOT / "data" / f"{name}.csv"))
            )

        users = read_csv("users", users_schema)
        purchases = read_csv("purchases", purchases_schema)
        products = read_csv("products", products_schema)
        for name, df, key in [
            ("users", users, "user_id"),
            ("purchases", purchases, "purchase_id"),
            ("products", products, "product_id"),
        ]:
            show_result(df.orderBy(key), f"p1_{name}", f"1. Read {name}.csv")

        # 2. Видалення рядків з будь-яким пропущеним значенням у кожному DF.
        before = [users.count(), purchases.count(), products.count()]
        users = users.dropna(how="any")
        purchases = purchases.dropna(how="any")
        products = products.dropna(how="any")
        after = [users.count(), purchases.count(), products.count()]
        cleaning = spark.createDataFrame(
            [(name, b, a, b - a) for name, b, a in
             zip(["users", "purchases", "products"], before, after)],
            ["dataset", "before", "after", "removed"],
        )
        show_result(cleaning, "p2_cleaning_summary", "2. Drop rows with missing values")
        for name, df, key in [
            ("users", users, "user_id"),
            ("purchases", purchases, "purchase_id"),
            ("products", products, "product_id"),
        ]:
            show_result(df.orderBy(key), f"p2_{name}", f"2. Cleaned {name}")

        # 3. Для загальної суми потрібні лише покупки й товари.
        # Не відкидаємо покупку через відсутність даних про вік/ім'я користувача.
        purchase_details = purchases.join(products, "product_id", "inner").withColumn(
            "purchase_amount", F.col("quantity") * F.col("price")
        )
        category_totals = purchase_details.groupBy("category").agg(
            F.sum("purchase_amount").alias("total_amount")
        ).orderBy(F.desc("total_amount"), "category")
        show_result(category_totals, "p3_category_totals", "3. Total spending by category")

        # 4. between(18, 25) включає обидві межі.
        young_purchases = purchase_details.join(
            users.filter(F.col("age").between(18, 25)).select("user_id"),
            "user_id", "inner",
        )
        young_totals = young_purchases.groupBy("category").agg(
            F.sum("purchase_amount").alias("total_amount_18_25")
        )
        show_result(
            young_totals.orderBy(F.desc("total_amount_18_25"), "category"),
            "p4_young_category_totals", "4. Spending by category, ages 18-25 inclusive",
        )

        # 5. Знаменник — загальна сума витрат саме вікової групи 18–25.
        total_young = young_totals.agg(
            F.sum("total_amount_18_25").alias("all_categories_amount_18_25")
        )
        shares = young_totals.crossJoin(total_young).withColumn(
            "spending_percentage",
            F.when(
                F.col("all_categories_amount_18_25") > 0,
                F.round(
                    F.col("total_amount_18_25")
                    / F.col("all_categories_amount_18_25") * 100, 2,
                ),
            ).otherwise(F.lit(None).cast("decimal(10,2)")),
        )
        show_result(shares.orderBy("category"), "p5_spending_shares", "5. Share of spending, ages 18-25 (%)")

        # 6. Спільний додатний знаменник: порядок сум = порядок точних часток.
        # Це зберігає правильний рейтинг навіть за однакових округлених відсотків.
        top3 = shares.orderBy(F.desc("total_amount_18_25"), "category").limit(3)
        show_result(top3, "p6_top3_categories", "6. Top 3 categories, ages 18-25 (%)")

        # Компактні CSV-результати для незалежної перевірки й повторного аналізу.
        import csv
        for filename, df in [
            ("p3_category_totals", category_totals),
            ("p4_young_category_totals", young_totals.orderBy("category")),
            ("p5_spending_shares", shares.orderBy("category")),
            ("p6_top3_categories", top3),
        ]:
            with (OUTPUT / f"{filename}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(df.columns)
                writer.writerows(df.collect())
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
