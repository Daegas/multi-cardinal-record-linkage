
from pathlib import Path
import re
import unicodedata

import pandas as pd


def prepare_splink_data(
    parquet_path: str | Path,
    *,
    date_columns: list[str] | None = None,
    sex_columns: list[str] | None = None,
    text_columns: list[str] | None = None,
    unique_id_column: str = "unique_id",
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Load and prepare a Parquet dataset for Splink.

    Operations:
        - Assign unique IDs when they do not already exist.
        - Standardize text (case, whitespace and special characters).
        - Convert empty strings and unknown-value indicators to nulls.
        - Standardize specified date columns to YYYY-MM-DD.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the input Parquet file.
    date_columns : list[str] | None
        Columns to convert to YYYY-MM-DD strings.
    text_columns : list[str] | None
        Columns to clean as text. If None, object and string
        columns are detected automatically.
    unique_id_column : str
        Name of the unique identifier column.
    dayfirst : bool
        Whether dates should be interpreted as day-first.

    Returns
    -------
    pd.DataFrame
        Prepared dataset for Splink.
    """

    # Load data
    df = pd.read_parquet(parquet_path).copy()

    date_columns = date_columns or []
    sex_columns = sex_columns or []

    # Validate date columns
    missing_dates = set(date_columns) - set(df.columns)
    if missing_dates:
        raise ValueError(f"Date columns not found: {missing_dates}")

    # Assign unique IDs
    if unique_id_column not in df.columns:
        df.insert(0, unique_id_column, range(1, len(df) + 1))
    else:
        if df[unique_id_column].isna().any():
            raise ValueError(f"{unique_id_column} contains null values.")
        if df[unique_id_column].duplicated().any():
            raise ValueError(f"{unique_id_column} contains duplicates.")

    # Unknown-value indicators
    null_values = {
        "", "na", "n/a", "nan", "none", "null",
        "unknown", "desconocido", "no especificado"
    }

    def clean_text(value):
        if pd.isna(value):
            return pd.NA

        value = (unicodedata.normalize("NFKD", value)
                    .encode("ASCII", "ignore")
                    .decode("utf-8")
                ) # Remove accents
        value = value.strip().casefold() #lowercase and strip whitespace

        if value in null_values:
            return pd.NA

        # Remove special characters
        value = re.sub(r"[^\w\s]", " ", value)

        # Normalize whitespace
        value = re.sub(r"\s+", " ", value).strip()


        return value if value else pd.NA

    # Detect text columns
    if text_columns is None:
        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

    # Clean text columns
    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"Text column not found: {col}")

        if col not in date_columns and col != unique_id_column:
            df[col] = df[col].map(clean_text).astype("string")

    # Standardize dates
    for col in date_columns:
        dates = pd.to_datetime(
            df[col],
            errors="coerce",
            format="mixed",
            dayfirst=dayfirst,
        )

        df[col] = dates.dt.strftime("%Y-%m-%d").astype("string")

    # Standardize sex columns
    for col in sex_columns:
        df[col] = df[col].replace({
            "hombre": "masculino",
            "mujer": "femenino",
            "varon": "masculino",
            "dama": "femenino",
            "m": "femenino",
            "h": "masculino",
            "f": "femenino",
            "femina": "femenino",
            "masculina": "masculino",
        })

    return df


ENTIDADES = {
    1: "aguascalientes",
    2: "baja california",
    3: "baja california sur",
    4: "campeche",
    5: "coahuila",
    6: "colima",
    7: "chiapas",
    8: "chihuahua",
    9: "ciudad de méxico",
    10: "durango",
    11: "guanajuato",
    12: "guerrero",
    13: "hidalgo",
    14: "jalisco",
    15: "estado de méxico",
    16: "michoacán",
    17: "morelos",
    18: "nayarit",
    19: "nuevo león",
    20: "oaxaca",
    21: "puebla",
    22: "querétaro",
    23: "quintana roo",
    24: "san luis potosí",
    25: "sinaloa",
    26: "sonora",
    27: "tabasco",
    28: "tamaulipas",
    29: "tlaxcala",
    30: "veracruz",
    31: "yucatán",
    32: "zacatecas",
}


ALIAS_ENTIDADES = {
    "coahuila de zaragoza": "coahuila",
    "veracruz de ignacio de la llave": "veracruz",
    "michoacán de ocampo": "michoacán",

    "cdmx": "ciudad de méxico",
    "cdmx.": "ciudad de méxico",
    "df": "ciudad de méxico",
    "d.f.": "ciudad de méxico",
    "distrito federal": "ciudad de méxico",

    "méxico": "estado de méxico",
    "edomex": "estado de méxico",
    "edo. de méxico": "estado de méxico",
    "estado de mexico": "estado de méxico",

    "nuevo leon": "nuevo león",
    "queretaro": "querétaro",
    "san luis potosi": "san luis potosí",
    "yucatan": "yucatán",
    "michoacan": "michoacán",
}


def _normalize_key(value: str) -> str:
    """Normalize text for comparison."""
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1, start=1):
        current = [i]

        for j, c2 in enumerate(s2, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (c1 != c2)

            current.append(min(insert, delete, substitute))

        previous = current

    return previous[-1]


def normalize_entities(
    df: pd.DataFrame,
    *,
    column: str,
    unique_id_column: str = "unique_id",
    max_distance: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize Mexican state names using official names, aliases
    and Levenshtein distance.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized entities.

    report : pd.DataFrame
        Changes sorted by Levenshtein distance (descending).
    """

    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    if unique_id_column not in df.columns:
        raise ValueError(f"Column not found: {unique_id_column}")

    if df[unique_id_column].isna().any():
        raise ValueError("Unique IDs cannot contain null values.")

    if df[unique_id_column].duplicated().any():
        raise ValueError("Unique IDs must be unique.")

    # Normalized official names
    official = {
        _normalize_key(name): name
        for name in ENTIDADES.values()
    }

    # Normalized aliases
    aliases = {
        _normalize_key(alias): official[
            _normalize_key(target)
        ]
        for alias, target in ALIAS_ENTIDADES.items()
    }

    official_keys = list(official.keys())

    changes = []

    def normalize_value(value, unique_id):

        if pd.isna(value):
            return value

        original = str(value)
        key = _normalize_key(original)

        distance = 0

        # Exact official name
        if key in official:
            normalized = official[key]

        # Known alias
        elif key in aliases:
            normalized = aliases[key]

        # Fuzzy matching
        else:
            distances = [
                (_levenshtein(key, candidate), candidate)
                for candidate in official_keys
            ]

            distance, best_candidate = min(
                distances,
                key=lambda x: x[0]
            )

            if distance <= max_distance:
                normalized = official[best_candidate]
            else:
                normalized = original
                distance = None

        if original != normalized:
            if distance is None:
                distance = _levenshtein(
                    _normalize_key(original),
                    _normalize_key(normalized)
                )

            changes.append({
                "unique_id": unique_id,
                "original": original,
                "normalized": normalized,
                "distance": distance,
            })

        return normalized

    df[column] = [
        normalize_value(value, uid)
        for value, uid in zip(
            df[column],
            df[unique_id_column]
        )
    ]

    # Changes report
    report = pd.DataFrame(
        changes,
        columns=[
            "unique_id",
            "original",
            "normalized",
            "distance",
        ]
    )

    report = report.sort_values(
        "distance",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    return df, report
