import random
import re

import tw_values
from tw_values import company_id_valid


def test_company_id_checksum_examples():
    # 04595257: products' digit sums total 30, divisible by 10 and 5.
    assert company_id_valid("04595257")
    # 7th digit is 7: 10458574 passes only through the +1 rule.
    assert company_id_valid("10458574")
    assert not company_id_valid("12345678")
    assert not company_id_valid("1234567")


def test_generated_values_match_their_formats():
    picker = random.Random(0)
    patterns = {
        tw_values.company_id: r"\d{8}",
        tw_values.passport: r"3\d{8}",
        tw_values.household_no: r"[A-Z]\d{7}",
        tw_values.national_id: r"[A-Z][12]\d{8}",
        tw_values.ptt_id: r"[A-Za-z][A-Za-z0-9]{1,11}",
        tw_values.line_id: r"[a-z0-9._-]{1,20}",
        tw_values.mobile: r"09\d{2}[- ]?\d{3}[- ]?\d{3}",
        tw_values.license_plate: r"[A-Z]{3}-\d{4}|[A-Z]{2}-\d{4}|\d{4}-[A-Z]{2}",
        tw_values.medical_license: r"(醫字第|醫師字第)?\d{5,6}號?",
        tw_values.military_id: r"[A-Z][12]\d{8}|軍字第\d{6,7}號",
        tw_values.personal_url: r"https://[\w.\-/@?=]+",
    }
    for generate, pattern in patterns.items():
        for _ in range(200):
            value = generate(picker)
            assert re.fullmatch(pattern, value), (generate.__name__, value)


def test_every_company_id_passes_its_own_check():
    picker = random.Random(1)
    assert all(company_id_valid(tw_values.company_id(picker)) for _ in range(500))
