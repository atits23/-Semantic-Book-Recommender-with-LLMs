"""Download the tutorial dataset; alternatively place books.csv in data/raw."""

import io
import shutil
import urllib.request
import zipfile

from book_recommender.config import RAW, make_directories

URL = "https://www.kaggle.com/api/v1/datasets/download/dylanjcastillo/7k-books-with-metadata"


def main():
    make_directories()
    if RAW.exists():
        print(f"Using existing file: {RAW}")
        return
    request = urllib.request.Request(
        URL, headers={"User-Agent": "BookRecommenderStudentProject/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        names = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "books.csv"]
        if len(names) != 1:
            raise ValueError("Expected exactly one books.csv in the archive")
        temp = RAW.with_suffix(".tmp")
        with archive.open(names[0]) as source, temp.open("wb") as target:
            shutil.copyfileobj(source, target)
        temp.replace(RAW)
        print(f"Saved {RAW}")
    except Exception as exc:
        raise SystemExit(
            "Automatic download failed. Download 7k Books from "
            "https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata "
            f"and put books.csv in data/raw/. Details: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
