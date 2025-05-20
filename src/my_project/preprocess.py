from my_project.config import REQUIRED_HEADERS

def uniform_headers(df):
    # for each required header, find which actual column it matches and rename
    mapping = {}
    for std, variants in REQUIRED_HEADERS.items():
        for v in variants:
            if v in df.columns:
                mapping[v] = std
                break
        else:
            raise KeyError(f"No column found for '{std}' among {variants}")
    return df.rename(columns=mapping)[list(REQUIRED_HEADERS.keys())]
