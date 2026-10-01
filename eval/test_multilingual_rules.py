"""Check that Dutch and Spanish charge labels land in the intended subcategory.

Each case is (charge label, expected subcategory). Most are real labels from the
data (UPS Netherlands, DHL / forwarders in Latin America); the ones marked
"synthetic" are common invoice wordings not seen yet, kept so the patterns stay
correct when they do show up. Also covers the English labels the Dutch/Spanish
patterns must not disturb.

No database access needed. Run from the repo root:
    python eval/test_multilingual_rules.py
Exits non-zero if any case fails.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from taxonomy import rule_classify_row  # noqa: E402

CASES = [
    # --- Dutch (real labels) ---
    ("Brandstof toeslag", "Domestic Fuel Surcharge"),
    ("BRANDSTOFTOESLAG", "Domestic Fuel Surcharge"),
    ("Retourzendingen Brandstof toeslag", "Domestic Fuel Surcharge"),
    ("Surge Fee - Commercieel", "Demand / Peak Surcharge"),
    ("Surge Fee - Particulier", "Demand / Peak Surcharge"),
    ("WINTERTOESLAG", "Demand / Peak Surcharge"),
    ("Residentieel", "Residential Delivery/Surcharge"),
    ("Residentieel Adjustment", "Residential Delivery/Surcharge"),
    ("Adrescorrectie TB Express Saver", "Address Correction"),
    ("Adrescorrectie Dom. Express", "Address Correction"),
    ("Toeslag afh. afgel. gebied-bestem.", "Delivery Area Surcharge (DAS)"),
    ("Toeslag afgelegen gebied", "Delivery Area Surcharge (DAS)"),
    ("Toeslag groot pakket", "Additional Handling"),
    ("Vraagtoeslag-Groot pakket", "Additional Handling"),
    ("Additionele afhandelingskosten", "Additional Handling"),
    ("Retourzendingen Vraagtoeslag-Extra afhandel", "Additional Handling"),
    ("Retourzendingen Vraagtoeslag-Over max", "Additional Handling"),
    ("Retourzendingen Maximale afmeting overschreden", "Additional Handling"),
    ("Laadklep", "Additional Handling"),
    ("ASSISTENTIE", "Additional Handling"),
    ("Verzendcorrectiekosten", "Billing Adjustment / Correction"),
    ("MANUELE AANPASSING", "Billing Adjustment / Correction"),
    ("TB Express Saver Aanpassing", "Billing Adjustment / Correction"),
    ("Btw", "VAT"),
    ("Invoerrecht", "Duty & Import Tax"),
    ("Retourzendingen Invoerrecht", "Duty & Import Tax"),
    ("Inklaringstoeslag", "Customs / Brokerage"),
    ("Internationale verwerkingskosten", "Customs / Brokerage"),
    ("Extra Tariefposten", "Customs / Brokerage"),
    ("Uitbetalingskosten", "Disbursement Fee"),
    ("MILIEU BIJDRAGE", "Sustainability / Carbon Fee"),
    ("Opslag kosten", "Package Handling / Storage"),
    ("Kosten service klantoplossing", "Customer Service / Resolution Fee"),
    ("Vervoer", "General / Mode Not Specified"),
    ("VRACHTKOSTEN", "General / Mode Not Specified"),
    ("Retourzendingen TB Standard", "International / Export / Import Freight"),
    ("Retourzendingen WW Express Saver", "International / Export / Import Freight"),
    ("Retourzendingen Label Print", "Returns / Print Label Fee"),
    ("Retourzendingen 3 UPS Ophaalpogingen", "Returns / Print Label Fee"),
    # --- Dutch (synthetic) ---
    ("Zaterdaglevering", "Saturday / After Hours / Holiday"),
    ("Piektoeslag", "Demand / Peak Surcharge"),
    ("Handtekening vereist", "Signature Required"),
    ("Wachttijd", "Wait Time / Mileage / Toll"),
    ("Bezorgpoging", "Delivery Attempt / Reconsignment"),
    ("Korting", "General Discount"),
    ("Creditnota", "Credit From Carrier"),
    ("Douanekosten", "Customs / Brokerage"),
    ("Accijns", "Duty & Import Tax"),
    ("Omzetbelasting", "VAT"),
    ("Verzekering", "Declared Value / Insurance"),
    ("Aangegeven waarde", "Declared Value / Insurance"),
    ("Voorschot invoerrechten", "Duty & Import Tax"),
    # --- Spanish (real labels) ---
    ("IVA", "VAT"),
    ("Mexico IVA Freight", "VAT"),
    ("GASTOS SUPLIDOS ADUANA", "Customs / Brokerage"),
    ("Tramite Aduana", "Customs / Brokerage"),
    # --- Spanish (synthetic) ---
    ("Recargo por combustible", "Domestic Fuel Surcharge"),
    ("Entrega en sabado", "Saturday / After Hours / Holiday"),
    ("Entrega residencial", "Residential Delivery/Surcharge"),
    ("Zona extendida", "Delivery Area Surcharge (DAS)"),
    ("Correccion de direccion", "Address Correction"),
    ("Tiempo de espera", "Wait Time / Mileage / Toll"),
    ("Peaje", "Wait Time / Mileage / Toll"),
    ("Descuento", "General Discount"),
    ("Nota de credito", "Credit From Carrier"),
    ("IGV", "VAT"),
    ("Arancel", "Duty & Import Tax"),
    ("Despacho aduanal", "Customs / Brokerage"),
    ("Impuestos", "Sales / General Tax"),
    ("Gastos suplidos", "Disbursement Fee"),
    ("Seguro de carga", "Declared Value / Insurance"),
    ("Almacenaje", "Package Handling / Storage"),
    ("Recoleccion", "Pickup Service"),
    ("Flete", "General / Mode Not Specified"),
    ("Devolucion etiqueta", "Returns / Print Label Fee"),
    # --- English labels the new patterns must keep (or now get) right ---
    ("Fuel Surcharge", "Domestic Fuel Surcharge"),
    ("Address Correction Fuel Surcharge", "Fuel Surcharge Correction"),
    ("Address Correction Ground", "Address Correction"),
    ("Shipping Charge Correction Ground", "Billing Adjustment / Correction"),
    ("Corrected BOL", "Billing Adjustment / Correction"),
    ("0004700 CORRECTION FEE PREPAID", "Billing Adjustment / Correction"),
    ("WEEKEND SURCHARGE", "Saturday / After Hours / Holiday"),
    ("Weekend Charge", "Saturday / After Hours / Holiday"),
    ("Additional Piece Fee", "Additional Handling"),
    ("Credit Memo", "Credit From Carrier"),
    ("ADVANCE AT DESTINATION", "Disbursement Fee"),
    ("TB Express Saver", "International / Export / Import Freight"),
    ("TB Standard Undeliverable Return", "Returns / Print Label Fee"),
    ("Dom. Express Saver", "General / Mode Not Specified"),
    ("UPS WorldEase Surge Fee - Com", "Demand / Peak Surcharge"),
    ("Cargo Screening Fee", "Security Surcharge"),
    ("PICKUP", "Pickup Service"),
    ("Freight", "General / Mode Not Specified"),
    ("Performance Pricing", "Earned Discount"),
    # Look like invoice totals but are the base freight charge (see PRIORITY_OVERRIDES)
    ("*** SUM *** WEIGHT & CHARGE", "General / Mode Not Specified"),
    ("**%* SUM *** WEIGHT & CHARGE", "General / Mode Not Specified"),
    ("TOTALS", "General / Mode Not Specified"),
    # Reviewed by hand (REVIEWED_LABELS) and the patterns added alongside them
    ("FLAT", "General / Mode Not Specified"),
    ("DEST DELIVERY", "General / Mode Not Specified"),
    ("FINAL DELIVERY", "General / Mode Not Specified"),
    ("78 00% OlSC", "General Discount"),
    ("ISS", "Security Surcharge"),
    ("1SS", "Security Surcharge"),
    ("Delivery Cartage", "Ground"),
    ("Large Package Surcharge Comm - Length", "Additional Handling"),
    ("Dest. Terminal Handling-Forwarder - Greater of (Min", "Additional Handling"),
    ("IMP/A TERMINAL HNDLG", "Additional Handling"),
    ("Shipping Charge Correction Large Package Surcharge - Length + Girth", "Billing Adjustment / Correction"),
    ("THC (Terminal Handling Charge) - Export", "Ocean Freight"),
    ("Pick Up Cartage", "Pickup Service"),
    ("Addl. Handling weight", "Additional Handling"),
    ("org. Temp. Control Handling", "Additional Handling"),
    ("Origin Handling Fee", "Additional Handling"),
    ("MILEAGE", "Wait Time / Mileage / Toll"),
    ("Missing PLD Fee", "Billing Adjustment / Correction"),
]

# (charge type, description, expected subcategory). The description is matched first, so a
# generic charge type ("Freight", "Accessorial") must not override what the description says,
# and ^ / $ anchored patterns must see the description itself. The type still helps when the
# description alone matches nothing.
TYPED_CASES = [
    ("Freight", "ITEM 116030 SUB:4 MACHINES c100", "LTL Freight (Commodity Line)"),
    ("Freight", "SK ACTUATORS", "LTL Freight (Commodity Line)"),
    ("Freight", "PLT NMFC 051080-06 Faucets; Bibcocks; ... 70", "LTL Freight (Commodity Line)"),
    ("Freight", "Freight Charges", "General / Mode Not Specified"),
    ("Freight", "This field is not yet available", "General / Mode Not Specified"),
    ("Accessorial", "TOTALS", "General / Mode Not Specified"),
    ("Accessorial", "Not Previously Billed Foo", "Billing Adjustment / Correction"),
    ("Accessorial", "Returns Label Fee", "Returns / Print Label Fee"),
]

# Meaning not settled yet -- must stay unmatched so it lands in charge_review.
UNMATCHED_CASES = ["Original Invoice Amount", "LEASE; REPO & WAIVER", "Service Charge"]

failures = []
for label, expected in CASES:
    got = rule_classify_row(label, label)
    if got is None or got[1] != expected:
        failures.append(f"  {label!r}: expected {expected!r}, got {got[1] if got else None!r}")
for ctype, desc, expected in TYPED_CASES:
    got = rule_classify_row(ctype, desc)
    if got is None or got[1] != expected:
        failures.append(f"  {ctype!r} / {desc!r}: expected {expected!r}, got {got[1] if got else None!r}")
for label in UNMATCHED_CASES:
    if rule_classify_row(label, label) is not None:
        failures.append(f"  {label!r}: expected no rule to match, got {rule_classify_row(label, label)!r}")

total = len(CASES) + len(TYPED_CASES) + len(UNMATCHED_CASES)
print(f"{total - len(failures)}/{total} cases pass")
if failures:
    print("\n".join(failures))
    sys.exit(1)
