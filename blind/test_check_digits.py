from check_digits import fixed_id, fixed_text, mainland_check, taiwan_check


def test_taiwan_check_digit_matches_a_known_valid_id():
    assert taiwan_check("A12345678") == "9"


def test_mainland_check_character_matches_a_known_valid_id():
    assert mainland_check("11010519491231002") == "X"


def test_a_taiwan_id_keeps_its_spacing_and_gets_a_valid_last_digit():
    assert fixed_id("A 1 2 3 4 5 6 7 8 0") == "A 1 2 3 4 5 6 7 8 9"


def test_a_mainland_id_gets_its_check_character():
    assert fixed_id("110105194912310021") == "11010519491231002X"


def test_an_unknown_shape_is_left_alone():
    assert fixed_id("12345") == "12345"


def test_only_national_id_marks_change():
    marked = "號碼[[national_id:A123456780]]，電話[[phone:0912345670]]"
    assert fixed_text(marked) == "號碼[[national_id:A123456789]]，電話[[phone:0912345670]]"
