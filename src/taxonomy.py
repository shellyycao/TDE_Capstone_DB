"""Charge category taxonomy — single source of truth for this pipeline.

Imported by glue_jobs/charge_type.py via Glue's --extra-py-files job
parameter, so the taxonomy only ever lives in one place. If you change a
category here, only charge_type.py needs to be re-run afterward (then
build_charge_categorized.py + dashboard.py to propagate it downstream).

Patterns are lowercase regex fragments matched against `Charge Type` and
`Charge Description` combined into one string (see normalize/rule_classify_row)
-- not tried separately field-by-field. An earlier version tried `Charge Type`
to completion first and fell back to `Charge Description` only if nothing
matched; that badly undercounted subcategories like "Ground", because a
generic `Charge Type` value like "Freight" would match a catch-all pattern
before ever looking at `Charge Description`, where the real detail (e.g.
"Ground Commercial") actually lives.

Within each major, order matters: more specific subcategories are listed before
generic ones (e.g. "Chargeback Fuel Surcharge" before the generic fuel pattern;
"Ground" before the generic "Transportation / Base Charge" fallback).
"""

import re

import pandas as pd

TAXONOMY = {
    "Fuel Surcharge": {
        "Fuel Surcharge Correction": [r"correction.*fuel", r"fuel.*correction"],
        "Chargeback Fuel Surcharge": [r"chargeback.*fuel"],
        "International Fuel Surcharge": [r"worldease.*fuel", r"international.*fuel"],
        "Domestic Fuel Surcharge": [r"\bfuel surcharge\b", r"\bfuel\b", r"brandstof", r"\bfsc\b"],
    },
    "Accessorial / Delivery Surcharge": {
        "Delivery Area Surcharge (DAS)": [r"\bdas\b", r"delivery area surcharge", r"afgel(egen)?\.? gebied", r"limited access", r"beyond area", r"extended area", r"metro charge", r"\barbitrary\b", r"high cost delivery", r"trade show", r"constr(uction|/utility)? site"],
        "Residential Delivery/Surcharge": [r"residential"],
        "Signature Required": [r"signature"],
        "Additional Handling": [r"add'?l handling", r"additional handling", r"handling.*dimension", r"lift ?gate", r"inside delivery", r"protect from freez", r"temp\.? controlled", r"excess l(e)?n(gth)?\b", r"non.?conveyable", r"extreme length", r"over ?dimension", r"\boversize\b", r"over ?(weight|length)\b", r"over dim\b", r"groot pakket", r"afhandeling", r"dangerous goods", r"\bhazardous\b"],
        "Saturday / After Hours / Holiday": [r"saturday", r"after hours", r"\bholiday\b"],
        "Demand / Peak Surcharge": [r"demand surcharge", r"surge emergency", r"\bpeak\b"],
        "Wait Time / Mileage / Toll": [r"wait time", r"additional miles", r"\btoll(ing)?\b", r"detention", r"layover", r"stop ?off", r"time on.?site"],
        "Address Correction": [r"address correction"],
        "Appointment / Notification": [r"\bappointment\b", r"hour notice", r"\bnotification\b"],
        "Delivery Attempt / Reconsignment": [r"\battempt(ed)?\b", r"reconsign", r"redelivery"],
        "Regulatory / Compliance Surcharge": [r"compliance", r"regulatory"],
        "Security Surcharge": [r"security surcharge", r"\bscreening\b", r"x-ray"],
        # Symmetric counterpart to "Residential Delivery/Surcharge". Deliberately narrow
        # (only "commercial", not "business") -- raw data has garbage commodity-description
        # rows like "Business or Office Machines, viz." that a bare `\bbusiness\b` pattern
        # would wrongly sweep in. Freight service-tier variants ("Ground Commercial",
        # "Next Day Air Commercial") are pulled out to Line Haul / Base Transportation before
        # this ever runs -- see PRIORITY_OVERRIDES.
        "Commercial / Business Delivery Surcharge": [r"\bcommercial\b"],
    },
    "Discounts": {
        "Earned Discount": [r"earned discount", r"performance pricing"],
        "Grace Discount": [r"grace discount"],
        "General Discount": [r"\bdiscount\b"],
    },
    "Credits": {
        "Credit From Carrier": [r"credit from carrier"],
    },
    "Taxes & Customs": {
        "VAT": [r"\bvat\b", r"\bbtw\b"],
        "GST / HST": [r"\bgst\b", r"\bhst\b"],
        "Duty & Import Tax": [r"dut(y|ies)", r"import fee", r"import tax", r"invoerrecht"],
        "Customs / Brokerage": [r"customs?", r"inbound processing", r"merchandise processing", r"other gov(ernmen)?t\.? (agency )?fees?", r"\bisf\b", r"\ba?ams fees?\b", r"\bacms\b", r"\bfda\b", r"health canada", r"inklaring", r"border crossing", r"\bpga\b", r"entry prep", r"entry fee", r"multiline entry", r"\bbroker\b", r"\bcci\b", r"\baes\b", r"\b[ao]xp declaration", r"\bbonded\b", r"brokerage", r"export declaration", r"international processing", r"internationale verwerkingskosten", r"\beei\b", r"commercial invoice", r"entry line"],
        "Sales / General Tax": [r"\btax(es)?\b", r"federal ?tax", r"provincial ?tax"],
    },
    "Administrative & Service Fees": {
        # Always caught earlier by PRIORITY_OVERRIDES; kept here too so it still shows up
        # as a documented subcategory and as a TF-IDF fallback reference candidate.
        "Chargeback / Reversal": [r"\bchargeback\b"],
        # Row-level check: 99.1% of the rows matching these patterns are POSITIVE charges
        # (e.g. "Shipping Charge Correction Ground" is 4,376 positive vs. 6 negative rows)
        # -- this is billing/accounting corrections that add to the invoice, not a price
        # concession (Discounts) or a carrier-issued refund (Credits).
        "Billing Adjustment / Correction": [r"billing adjustment", r"shipping charge correction", r"verzendcorrectiekosten", r"\brebill\b", r"reweigh", r"weight changed", r"\binspection\b"],
        "Disbursement Fee": [r"disbursement"],
        "Third Party Billing": [r"third party billing"],
        "Document Fee": [r"document fee", r"documents? preparation", r"air ?way ?bill", r"\bawb\b", r"manifest fee", r"admin docs", r"paperwork", r"bill of ?lading", r"delivery order", r"\bd/o fee\b"],
        "Returns / Print Label Fee": [r"print.*label", r"returns print label", r"return to sender"],
        "Declared Value / Insurance": [r"declared value", r"\binsurance\b", r"extended liability", r"value protection"],
        "Cancellation Fee": [r"cancell?ation fee"],
        "Sustainability / Carbon Fee": [r"gogreen", r"carbon reduced", r"environmental fee"],
        "Package Handling / Storage": [r"package handling", r"warehouse storage", r"\bstorage\b", r"palleti[sz]"],
        "Pickup Service": [r"pick ?up"],
    },
    # Ground, Air, International, and Ocean Freight are *mode* subcategories -- the raw
    # label told us which mode. "General / Mode Not Specified" is NOT a fifth mode; it's
    # the fallback for when the raw label just says "Freight" / "Base" / "Transportation
    # Charge" / "Line Haul" without naming a mode at all.
    "Line Haul / Base Transportation": {
        "Ground": [r"\bground\b", r"\bltl\b", r"truckload", r"road ?(freight|transport)", r"delivery via truck"],
        # Broadened beyond just the named express tiers to also catch generic "air
        # freight"/"airfreight" labels and airport-transfer legs (incl. "AFS AIR FREIGHT
        # SURCHARGE ...", "Airline Airfreight Surcharge", "AIRPORT TRANSFER"). "rport
        # transfer" (not "airport transfer") also catches an OCR-mangled variant seen in
        # the raw data: "Al RPORT TRANSFER E".
        "Air Freight": [r"next day", r"2nd day", r"two day", r"3 day", r"third day", r"second day", r"\bair ?freight\b", r"rport transfer", r"airport fees?"],
        "International / Export / Import Freight": [r"worldwide express", r"ww express", r"\bexport\b", r"\bimport\b", r"world ?ease", r"international freight", r"\bww (saver|expedited)\b", r"worldwide (saver|expedited)", r"\bpremium \d", r"standard to canada"],
        "Ocean Freight": [r"ocean"],
        "General / Mode Not Specified": [r"line ?haul", r"transportation charge", r"\bbase\b", r"\bfreight\b", r"frt freight", r"\bvracht", r"all in rate", r"\bvervoer\b", r"deficit (wgt|weight)"],
        # Not a fee label at all: LTL freight bills put the *commodity description* of the
        # shipped goods ("FILM OR SHEETING NOI NMFC 156830-07 65", "Faucets; Bibcocks; Gate
        # Valves or", "PALLET STEEL EXPANSION TANK ... 150.0") on the line that carries the
        # base freight charge. These were ~83% of the uncategorized dollar value. Kept as
        # its own subcategory (not folded into "Ground") so the team can see how much of
        # Line Haul comes from these lines and decide separately how to report them.
        # Listed last so every real fee pattern above wins first, and excluded from the
        # TF-IDF fallback corpus (see TFIDF_EXCLUDE) so its generic nouns never pull
        # unrelated fee labels in via fuzzy matching.
        "LTL Freight (Commodity Line)": [
            # NMFC tariff vocabulary: item numbers, "not otherwise indicated", density/class
            r"\bnmfc\b", r"\bnoi\b", r"\bdensit(y|ies)\b", r"\bviz\b", r"\bclass ?\d{2,3}\b", r"\bcl ?\d{2,3}\b",
            r"\bitem \d{4,6}\b", r"\bsub: ?\d", r"\b\d{5,6}-\d{1,2}\b",
            # handling-unit prefixes, piece/weight columns, dimensions, UN hazmat numbers
            r"^(pallets?|plt|piece|pcs|skids?|skd|box|pails?)\b", r"\d;\d{3} \d{3}", r"\b\d+ ?x ?\d+ ?x ?\d+\b", r"\bun\d{4}\b",
            # high-dollar commodity nouns with none of the markers above (incl. misspellings seen in the data)
            r"\bartic?l(es|ces)\b", r"\bac(tu|cu|c)ators\b", r"\bmachine(s|ry)\b", r"\bappliances\b", r"\bfaucets\b",
            r"cable or wire", r"wires ropes", r"\bmodems?\b", r"\belectronics\b", r"vinyl records", r"\bfiberboard\b", r"\bwood doors\b",
        ],
    },
}

CATCH_ALL = ("Other / Uncategorized", "Unclassified")

# Checked BEFORE the main taxonomy loop -- these are cases where a generic pattern
# elsewhere in TAXONOMY would otherwise fire first and give the wrong answer.
PRIORITY_OVERRIDES = [
    # Fuel-specific chargebacks must win over the generic chargeback override below.
    (r"chargeback.*fuel", ("Fuel Surcharge", "Chargeback Fuel Surcharge")),
    # A chargeback/reversal is classified by what kind of billing event it is, not by
    # what the underlying charge was for -- so this must outrank Line Haul / Base
    # Transportation's mode patterns and Accessorial's "commercial"/"residential"
    # patterns for things like "Chargeback Ground Commercial".
    (r"\bchargeback\b", ("Administrative & Service Fees", "Chargeback / Reversal")),
    # "Commercial Invoice" is a customs document, not a delivery surcharge -- must be
    # pulled out before Accessorial's "Commercial / Business Delivery Surcharge" pattern.
    (r"commercial invoice", ("Taxes & Customs", "Customs / Brokerage")),
    # Container-move charges (chassis rental, drayage, devanning at a container freight
    # station, container per-diem). Override rather than TAXONOMY pattern for the same
    # reason as "transport to port" below: "container"/"charges" are too generic for the
    # TF-IDF corpus.
    (r"\bchassis\b|drayage|\bdevan\b|\bcfs\b|per diem|\bthc\b|wharfage", ("Line Haul / Base Transportation", "Ocean Freight")),
    # "Ground Residential" / "Next Day Air Commercial" etc. are freight *service-tier*
    # names (the residential- or commercial-rate version of that service), not a
    # standalone residential/commercial delivery surcharge line -- without this override
    # they'd be caught by Accessorial's generic "residential"/"commercial" patterns
    # before Line Haul / Base Transportation is ever reached.
    (r"\bground\b.*\bresidential\b", ("Line Haul / Base Transportation", "Ground")),
    (r"\bground\b.*\bcommercial\b", ("Line Haul / Base Transportation", "Ground")),
    (r"\b(next day|2nd day|second day|3 day|third day)\b.*\bresidential\b", ("Line Haul / Base Transportation", "Air Freight")),
    (r"\b(next day|2nd day|second day|3 day|third day)\b.*\bcommercial\b", ("Line Haul / Base Transportation", "Air Freight")),
    # "Transport to port" -> Ocean Freight. This is an override rather than a TAXONOMY
    # pattern on purpose: TF-IDF reference docs are built FROM the TAXONOMY patterns (see
    # charge_type.py), so putting a phrase containing the generic word "transport" there
    # would leak "transport" into Ocean Freight's fallback reference doc -- which then
    # fuzzy-matches unrelated "Road transport" / "Dedicated transport" labels into Ocean
    # Freight too (found this by checking a re-run's output, not hypothetically -- it
    # actually happened). Overrides never feed the TF-IDF corpus, so this stays a
    # precise, narrow rule without that side effect.
    (r"transport to port", ("Line Haul / Base Transportation", "Ocean Freight")),
]

# Checked AFTER the main taxonomy loop, only when nothing in TAXONOMY matched -- the
# opposite of PRIORITY_OVERRIDES. For prefixed labels the underlying charge wins when it
# is recognizable ("Returns Fuel Surcharge" stays Fuel Surcharge, "Not Previously Billed
# Residential Surcharge" stays Residential); the prefix only decides the category when
# the rest of the label is something the taxonomy doesn't know ("Returns Pre-Release
# Notification Surcharge", "Not Previously Billed Missing PLD Fee").
# "Retourzendingen" is Dutch for returns. Kept out of TAXONOMY so these very common
# prefixes never enter the TF-IDF corpus either.
FALLBACK_RULES = [
    (r"^not previously billed\b", ("Administrative & Service Fees", "Billing Adjustment / Correction")),
    (r"^(retourzendingen|returns)\b", ("Administrative & Service Fees", "Returns / Print Label Fee")),
]

# Subcategories excluded from the TF-IDF fallback reference corpus (see ref_docs
# construction in charge_type.py). Checked empirically, not hypothetically: every
# literal "chargeback" mention in the data is already caught exactly by the
# `\bchargeback\b` override/rule above (there are zero "charge back" / "charge-back"
# spelling variants in the raw data that would need fuzzy matching to catch), so this
# subcategory's reference doc -- just the single word "chargeback" -- had no true
# positives to contribute via fallback. What it DID contribute: because "chargeback"
# shares its first 6 characters with the extremely common substring "charge" (present in
# hundreds of unrelated raw labels -- "Service Charge", "Weekend Charge", "Order Lane
# Charge", "Terminal Charges", ...), character n-gram TF-IDF scored many of them similar
# enough to fuzzy-match into "Chargeback / Reversal" -- a 100% false-positive rate across
# the rows it had pulled in this way. Excluding it from the candidate pool removes that
# failure mode entirely without weakening real chargeback detection at all, since the
# exact-match path was already exhaustive.
TFIDF_EXCLUDE = {
    ("Administrative & Service Fees", "Chargeback / Reversal"),
    # Same failure mode, same fix: this reference doc is "chargeback fuel", which still
    # shares "charge" with the same flood of generic "___ Charge" labels.
    # `chargeback.*fuel` in PRIORITY_OVERRIDES is exhaustive for real fuel chargebacks,
    # so nothing real is lost.
    ("Fuel Surcharge", "Chargeback Fuel Surcharge"),
    # Commodity descriptions, not fee names -- see the comment on this subcategory.
    ("Line Haul / Base Transportation", "LTL Freight (Commodity Line)"),
}


def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def rule_classify_row(charge_type, charge_desc):
    t = (normalize(charge_type) + " " + normalize(charge_desc)).strip()
    if not t:
        return None
    for pat, result in PRIORITY_OVERRIDES:
        if re.search(pat, t):
            return result
    for major, subcats in TAXONOMY.items():
        for sub, patterns in subcats.items():
            for pat in patterns:
                if re.search(pat, t):
                    return (major, sub)
    for pat, result in FALLBACK_RULES:
        if re.search(pat, t):
            return result
    return None
