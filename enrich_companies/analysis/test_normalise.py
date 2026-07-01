"""Quick regression harness for normalise().  Run: python test_normalise.py"""
import importlib.util

spec = importlib.util.spec_from_file_location("m", "bt_stp1b_ch_match.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
normalise = m.normalise

CASES = {
    # period / TLD handling
    "WHATS.ON BRIGHTON LTD": "whatson brighton",
    "Whatson": "whatson",
    "acme.com": "acme",
    "acme.co.uk": "acme",
    "commercial services": "commercial services",   # no false TLD strip
    # dotted legal suffix
    "WORK WORKS TRAINING SOLUTIONS C.I.C.": "work works training solutions",
    "Work Works Training Solutions CIC": "work works training solutions",
    "L.S.I. LTD": "lsi",
    "LSI": "lsi",
    # leading article, gated so initialisms survive
    "A MILLION VOICES LTD": "million voices",
    "Million Voices Ltd": "million voices",
    "A B C LTD": "abc",
    # descriptor stripping symmetry
    "Next Group Plc": "next",
    "NEXT GROUP PLC": "next",
    # 'of' connector
    "Budget Appliances Of Beckenham Limited": "budget appliances beckenham",
}


def main():
    ok = True
    for raw, want in CASES.items():
        got = normalise(raw)[0]
        flag = "OK " if got == want else "XX "
        if got != want:
            ok = False
        print(f"{flag}{raw!r:45} -> {got!r:35} want {want!r}")
    print("ALL PASS" if ok else "FAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
