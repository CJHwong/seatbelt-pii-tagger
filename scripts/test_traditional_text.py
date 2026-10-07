from traditional_text import is_traditional


def test_is_traditional_needs_traditional_markers_and_no_simplified_ones():
    assert is_traditional("這個國家的電話號碼與地址都寫在門口，我們會對照。")
    assert not is_traditional("这个国家的电话号码与地址都写在门口，我们会对照。")
    assert not is_traditional("這個国家的電話")
    assert not is_traditional("東京 Tokyo")
