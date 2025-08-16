import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # === Load Data ===
    df = pd.read_csv("data/processed/articles_clean.csv")

    print("=== Basic Info ===")
    print(df.info())
    print("\n=== First Rows ===")
    print(df.head())

    # === 1. Article Length Distribution ===
    df["char_count"] = df["text"].astype(str).apply(len)
    print("\n=== Article Length (chars) ===")
    print(df["char_count"].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(df["char_count"], bins=30, kde=True)
    plt.title("Distribution of Article Lengths")
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # === 2. Articles per Domain ===
    if "url" in df.columns:
        df["domain"] = df["url"].str.extract(r"https?://([^/]+)/")
        domain_counts = df["domain"].value_counts()
        print("\n=== Top Domains ===")
        print(domain_counts.head())

        plt.figure(figsize=(8,5))
        domain_counts.head(10).plot(kind="bar")
        plt.title("Top 10 Domains")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # === 3. Publication Dates (if available) ===
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df["date"] = df["published"].dt.date
        date_counts = df["date"].value_counts().sort_index()
        print("\n=== Articles per Day ===")
        print(date_counts.tail())

        plt.figure(figsize=(10,4))
        date_counts.plot()
        plt.title("Articles over Time")
        plt.ylabel("Count")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
