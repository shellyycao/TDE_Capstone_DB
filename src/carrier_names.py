"""Carrier code -> display name mapping.

The Data Import export's "Carrier Name" column is actually full of short
internal/SCAC-style codes ("FEDB", "XPOL", "ODFL", ...), not human-readable
carrier names -- see the "Carry name" investigation this was built from.

CARRIER_NAMES below only has entries I could identify with reasonable
confidence (mostly real, well-known LTL/parcel carriers whose codes are
unambiguous industry abbreviations). Everything else is deliberately left
unmapped rather than guessed -- a wrong guess (e.g. inventing a company
name for an internal code like "BTX" or "PYLE") is worse than leaving it
as the raw code, since it would look authoritative on the dashboard while
being fabricated.

display_name() falls back to the raw code for anything not in the map, so
nothing breaks or disappears if a code isn't recognized -- it just isn't
prettified yet.
"""

CARRIER_NAMES = {
    "UPS": "UPS",
    "DHL": "DHL Express",
    "DHLE": "DHL Express",
    "ODFL": "Old Dominion Freight Line",
    "ARCB": "ArcBest / ABF Freight",
    "SEFL": "Southeastern Freight Lines",
    "SCHN": "Schneider National",
    "ESTE": "Estes Express Lines",
    "XPOL": "XPO Logistics",
}

# Codes seen in the actual data that aren't in CARRIER_NAMES above --
# tracked here (rather than silently falling through) so it's obvious at a
# glance what still needs a real name from someone who knows this system's
# conventions, instead of that information only being visible by re-running
# the coverage check.
UNRESOLVED_CODES = {
    "AIRS", "AIT", "ATSI", "AXIS", "BTX", "BUFF", "CHRO", "CORD",
    "CRAN-I", "CRAN-V", "DHLF-V", "DISH", "DUPR", "EDGE", "EXP-C",
    "EXP-CR", "EXP-I", "EXP-V", "FDFB", "FEDB", "FEDC", "FEDCA", "FEDM",
    "FEDN", "FFWD", "FITZ", "GEOD", "GLOT", "ICAT", "LOAD", "MACO",
    "MVLO", "NEWL", "OAKV", "PACE", "PEGA", "PEGA-CR", "PRI1", "PROD",
    "PYLE", "RCOU", "RIVE", "RLCA", "SBA", "SEAC", "SPDL", "TDEP",
    "TFRC", "TRAN", "UPSM", "VCK", "XPOF",
}


def display_name(code):
    """Human-readable carrier name for a raw carrier code, or the code
    itself if it isn't in CARRIER_NAMES yet."""
    if not code:
        return code
    return CARRIER_NAMES.get(code.strip(), code.strip())
