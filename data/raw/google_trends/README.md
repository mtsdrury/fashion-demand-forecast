# Google Trends Data

## How to Download

1. Go to [Google Trends](https://trends.google.com/trends/)
2. For each term below, search with these settings:
   - **Region:** United States
   - **Time range:** 2004-01-01 to present
   - **Category:** All categories
   - **Search type:** Web Search
3. Click the download button (arrow icon) to export as CSV
4. Save each file with the naming convention: `{term}.csv` (use underscores for spaces)

## Terms

| Search Term          | Filename                  |
|----------------------|---------------------------|
| athleisure           | athleisure.csv            |
| fast fashion         | fast_fashion.csv          |
| streetwear           | streetwear.csv            |
| sustainable fashion  | sustainable_fashion.csv   |
| luxury fashion       | luxury_fashion.csv        |

## File Format

Google Trends CSVs have a header section (first 2 lines) followed by the data:

```
Category: All categories

Month,{term}: (United States)
2004-01,0
2004-02,0
...
```

The data loader (`src/data_loader.py`) handles this format automatically.
