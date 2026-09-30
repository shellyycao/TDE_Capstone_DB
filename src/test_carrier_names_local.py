"""Local, read-only check of carrier_names.py's coverage against the real
data_import_lookup table in S3. Doesn't touch the pipeline or write
anything -- just reports how much of the actual data is covered and
which codes still need a real name.

Usage:
  python src/test_carrier_names_local.py
"""

import csv
import io

import boto3

from carrier_names import CARRIER_NAMES, UNRESOLVED_CODES, display_name

BUCKET = "capstone-tde-demo"
KEY = "curated/data_import_lookup/data_import_lookup.csv"

s3 = boto3.client("s3", region_name="us-east-1")
obj = s3.get_object(Bucket=BUCKET, Key=KEY)
body = obj["Body"].read().decode("utf-8", errors="replace")
rows = list(csv.DictReader(io.StringIO(body)))

codes_in_data = {r["carrier_name"].strip() for r in rows if r["carrier_name"].strip()}
mapped = codes_in_data & CARRIER_NAMES.keys()
unmapped = codes_in_data - CARRIER_NAMES.keys()

row_count_mapped = sum(1 for r in rows if r["carrier_name"].strip() in CARRIER_NAMES)

print(f"{len(rows)} total import-batch rows, {len(codes_in_data)} distinct carrier codes")
print(f"{len(mapped)}/{len(codes_in_data)} distinct codes mapped "
      f"({row_count_mapped}/{len(rows)} rows, {row_count_mapped / len(rows) * 100:.1f}%)")
print()

print("Mapped:")
for code in sorted(mapped):
    print(f"  {code:10s} -> {display_name(code)}")

print()
stale = UNRESOLVED_CODES - codes_in_data
if stale:
    print(f"Note: UNRESOLVED_CODES has {len(stale)} code(s) not present in current data: {sorted(stale)}")

new_unmapped = unmapped - UNRESOLVED_CODES
if new_unmapped:
    print(f"Note: {len(new_unmapped)} code(s) in current data not yet tracked in UNRESOLVED_CODES: {sorted(new_unmapped)}")

print()
print(f"Still unmapped ({len(unmapped)}):")
for code in sorted(unmapped):
    print(f"  {code}")
