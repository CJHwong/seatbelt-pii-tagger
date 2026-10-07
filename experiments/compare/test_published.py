from published import KIJI_LABELS, PII_300K_LABELS, kiji_row, multiconer_rows, names_row, pii_300k_row


def test_300k_name_parts_merge_and_a_birth_date_is_a_date():
    text = "Ann Lee, born 1990-02-03, logs in at 10:30 from 10.0.0.1."
    spans = [
        {"start": 0, "end": 3, "label": "GIVENNAME1"},
        {"start": 4, "end": 7, "label": "LASTNAME1"},
        {"start": 14, "end": 24, "label": "BOD"},
        {"start": 37, "end": 42, "label": "TIME"},
        {"start": 48, "end": 56, "label": "IP"},
    ]
    row = pii_300k_row(text, spans)
    assert [(text[e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [
        ("Ann Lee", "person"),
        ("1990-02-03", "date"),
    ]
    assert [text[d["start"] : d["end"]] for d in row["dont_care"]] == ["10.0.0.1"]
    assert row["kind"] == "typed"


def test_300k_street_with_city_is_an_address_and_a_lone_city_stays_unmarked():
    text = "Ship to 12 Oak Street, Springfield. Visit Denver soon."
    spans = [
        {"start": 8, "end": 10, "label": "BUILDING"},
        {"start": 11, "end": 21, "label": "STREET"},
        {"start": 23, "end": 34, "label": "CITY"},
        {"start": 42, "end": 48, "label": "CITY"},
    ]
    row = pii_300k_row(text, spans)
    assert [(text[e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [
        ("12 Oak Street, Springfield", "address"),
    ]
    assert row["dont_care"] == []


def test_300k_map_covers_every_label_in_the_file():
    seen = {
        "TIME", "USERNAME", "IDCARD", "EMAIL", "SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE", "LASTNAME1", "BOD",
        "IP", "GIVENNAME1", "CITY", "SEX", "STATE", "TEL", "BUILDING", "TITLE", "STREET", "POSTCODE", "DATE",
        "PASS", "COUNTRY", "SECADDRESS", "LASTNAME2", "GIVENNAME2", "GEOCOORD", "LASTNAME3", "CARDISSUER",
    }  # fmt: skip
    assert seen <= set(PII_300K_LABELS)


def test_names_row_joins_latin_tokens_with_spaces_and_keeps_only_persons():
    row = names_row(["Ann", "Lee", "visited", "Paris"], [1, 2, 0, 5], " ")
    assert [(row["text"][e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [("Ann Lee", "person")]
    assert row["kind"] == "recall"


def test_names_row_joins_han_characters_without_spaces():
    row = names_row(["王", "小", "明", "在", "台", "北"], [1, 2, 2, 0, 5, 6], "")
    assert row["text"] == "王小明在台北"
    assert [(row["text"][e["start"] : e["end"]]) for e in row["entities"]] == ["王小明"]


def test_kiji_names_merge_a_plate_stays_typed_and_a_tax_number_is_dont_care():
    text = "Ann Lee drives plate ABC123 and files tax number 12-3456789."
    spans = [
        {"start": 0, "end": 3, "label": "FIRSTNAME", "value": "Ann"},
        {"start": 4, "end": 7, "label": "SURNAME", "value": "Lee"},
        {"start": 21, "end": 27, "label": "LICENSEPLATENUM", "value": "ABC123"},
        {"start": 49, "end": 59, "label": "TAXNUM", "value": "12-3456789"},
    ]
    row = kiji_row(text, spans)
    assert [(text[e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [
        ("Ann Lee", "person"),
        ("ABC123", "license_plate"),
    ]
    assert [text[d["start"] : d["end"]] for d in row["dont_care"]] == ["12-3456789"]


def test_kiji_map_covers_every_english_test_label():
    seen = {
        "FIRSTNAME", "SURNAME", "CITY", "BUILDINGNUM", "STREET", "STATE", "ZIP", "COMPANYNAME", "EMAIL",
        "PHONENUMBER", "URL", "COUNTRY", "DATEOFBIRTH", "DRIVERLICENSENUM", "LICENSEPLATENUM", "SECURITYTOKEN",
        "AGE", "PASSPORTID", "SSN", "PASSWORD", "IBAN", "IDCARDNUM", "TAXNUM", "NATIONALID", "CREDITCARDNUMBER",
        "USERNAME", "TITLE", "ORGANIZATION", "REGION",
    }  # fmt: skip
    assert seen <= set(KIJI_LABELS)


def test_multiconer_keeps_every_person_class_and_drops_the_rest():
    lines = [
        "# id 1\tdomain=en", "Ann _ _ B-Politician", "Lee _ _ I-Politician", "met _ _ O",
        "Bo _ _ B-OtherPER", "in _ _ O", "Paris _ _ B-HumanSettlement", "",
        "# id 2\tdomain=en", "Cy _ _ B-Athlete", "",
    ]  # fmt: skip
    rows = multiconer_rows(lines, " ")
    assert [[row["text"][e["start"] : e["end"]] for e in row["entities"]] for row in rows] == [
        ["Ann Lee", "Bo"],
        ["Cy"],
    ]
    assert {row["kind"] for row in rows} == {"recall"}
