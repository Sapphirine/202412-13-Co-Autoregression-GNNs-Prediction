import pandas
import yfinance  # type: ignore

tickers = (
    *("MSFT", "AAPL", "GOOG", "AMZN", "META", "NVDA", "ORCL", "ADBE", "AMD"),
    *("INTC", "IBM", "CRM", "TSM", "TSLA", "F", "GM", "LCID", "NIO", "BYDDF"),
    *("JPM", "GS", "MS", "MA", "PYPL", "KO", "PEP", "PG", "UL", "MCD", "SBUX"),
    *("NKE", "DIS", "BP", "JNJ", "PFE", "MRNA", "CAT", "BA", "GE"),
)

df = yfinance.download(tickers, interval="1mo", period="max")["Volume"]  # type: ignore
df = df.dropna(axis=1, how="all")  # type: ignore
df = df.dropna()  # type: ignore
df = pandas.concat(
    (
        pandas.DataFrame(
            {
                "CLASS_ID": [k] * df.shape[0],
                "CHANNEL_ID": [90.0] * df.shape[0],
                "YEAR": [x.year for x in df.index],  # type: ignore
                "MONTH": [x.month for x in df.index],  # type: ignore
                "UNITS": list(df[k]),  # type: ignore
            }
        )
        for k in df.columns  # type: ignore
    )
)
df.to_csv("yf.csv", index=False)
print(df)
