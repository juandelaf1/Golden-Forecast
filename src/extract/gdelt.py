import time
import requests
import pandas as pd

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def download_year(year):

    start = f"{year}0101000000"
    end = f"{year}1231235959"

    params = {
        "query": "gold",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 50,
        "startdatetime": start,
        "enddatetime": end
    }

    while True:
        response = requests.get(BASE_URL, params=params, timeout=30)

        if response.status_code == 429:
            print("429 recibido. Esperando 5 segundos...")
            time.sleep(5)
            continue

        response.raise_for_status()
        break

    print(response.headers.get("Content-Type"))
    print(response.text[:500])   # solo los primeros 500 caracteres

    data = response.json()

    if "articles" not in data:
        return pd.DataFrame()

    rows = []

    for article in data["articles"]:
        rows.append({
            "Date": article["seendate"][:8],
            "title": article["title"],
            "language": article["language"],
            "country": article["sourcecountry"],
            "url": article["url"]
        })

    df = pd.DataFrame(rows)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def download_gdelt_news():

    all_news = []

    for year in range(2015, 2026):

        print(f"Descargando noticias de {year}...")

        df = download_year(year)

        all_news.append(df)

        print(f"   {len(df)} noticias")

        # Esperar 5 segundos antes de la siguiente petición
        time.sleep(5)

    return pd.concat(all_news, ignore_index=True)


if __name__ == "__main__":

    df = download_gdelt_news()

    df.to_csv("data/raw/gold_news.csv", index=False)

    print(df.head())

    print(f"\nTotal noticias: {len(df)}")