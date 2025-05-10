# filters.py
import re

def filter_by_price(df, query):
    query = query.lower()
    between = re.search(r"between\s*₹?(\d+)\s*(?:and|-)\s*₹?(\d+)", query) or \
              re.search(r"range\s*₹?(\d+)\s*(?:to|-)\s*₹?(\d+)", query)
    above = re.search(r"(?:above|over|greater than|more than)\s*₹?(\d+)", query)
    below = re.search(r"(?:below|under|less than)\s*₹?(\d+)", query)

    if between:
        low, high = int(between.group(1)), int(between.group(2))
        return df[(df["Price"].astype(float) >= low) & (df["Price"].astype(float) <= high)]
    elif above:
        value = int(above.group(1))
        return df[df["Price"].astype(float) > value]
    elif below:
        value = int(below.group(1))
        return df[df["Price"].astype(float) < value]
    return df

def filter_by_specifications(df, query):
    query = query.lower()
    if "16 gb ram" in query or "16gb ram" in query:
        df = df[df["RAM"].str.contains("16", case=False, na=False)]
    if "8 gb ram" in query or "8gb ram" in query:
        df = df[df["RAM"].str.contains("8", case=False, na=False)]
    if "512 gb ssd" in query:
        df = df[df["Storage"].str.contains("512", case=False, na=False)]
    if "1 tb" in query:
        df = df[df["Storage"].str.contains("1", case=False, na=False)]
    if "i7" in query:
        df = df[df["Processor"].str.contains("i7", case=False, na=False)]
    if "i5" in query:
        df = df[df["Processor"].str.contains("i5", case=False, na=False)]
    if "ryzen 7" in query:
        df = df[df["Processor"].str.contains("ryzen 7", case=False, na=False)]
    if "ryzen 5" in query:
        df = df[df["Processor"].str.contains("ryzen 5", case=False, na=False)]
    return df

def filter_by_purpose(df, query):
    query = query.lower()

    if "gaming" in query:
        return df[
            df["Processor"].str.contains("i7|i9|ryzen 7|ryzen 9", case=False, na=False) &
            df["RAM"].str.contains("16|32", case=False, na=False) &
            df["Specifications"].str.contains("graphics", case=False, na=False)
        ]
    elif "office" in query:
        return df[
            df["Specifications"].str.contains("battery|webcam|office|zoom|teams", case=False, na=False) &
            df["RAM"].str.contains("8|16", case=False, na=False)
        ]
    elif "student" in query:
        return df[df["Price"].astype(float) < 50000]
    elif "remote" in query or "remote work" in query:
        return df[df["Specifications"].str.contains("battery|portable", case=False, na=False)]

    return df

