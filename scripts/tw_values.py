"""Value generators for Taiwanese fields. Each takes a random.Random and returns a string.

Faker's zh_TW locale is reused where it is right (names, the national ID with its
checksum, addresses, companies). The rest is built here because Faker lacks it or
gets it wrong for Taiwan (bank accounts come out as British IBANs). Format sources
are cited next to each generator.
"""

import random
import string

from faker import Faker

_FAKER = Faker("zh_TW")


def _seeded(picker: random.Random) -> Faker:
    _FAKER.seed_instance(picker.getrandbits(32))
    return _FAKER


def person(picker: random.Random) -> str:
    return _seeded(picker).name()


def surname(picker: random.Random) -> str:
    return person(picker)[0]


def national_id(picker: random.Random) -> str:
    # Faker's zh_TW ssn() implements the letter table and check digit.
    return _seeded(picker).ssn()


def address(picker: random.Random) -> str:
    return _seeded(picker).address()


def company(picker: random.Random) -> str:
    return _seeded(picker).company()


def city(picker: random.Random) -> str:
    return _seeded(picker).city()


def dob(picker: random.Random) -> str:
    born = _seeded(picker).date_of_birth(minimum_age=1, maximum_age=90)
    style = picker.randrange(4)
    if style == 0:
        return born.isoformat()
    if style == 1:
        return f"{born.year}/{born.month:02d}/{born.day:02d}"
    if style == 2:
        return f"{born.year}年{born.month}月{born.day}日"
    return f"民國{born.year - 1911}年{born.month}月{born.day}日"


def other_date(picker: random.Random) -> str:
    day = _seeded(picker).date_between(start_date="-2y", end_date="+1y")
    return f"{day.month}月{day.day}日" if picker.random() < 0.5 else f"{day.month}/{day.day}"


def mobile(picker: random.Random) -> str:
    digits = "09" + "".join(picker.choice(string.digits) for _ in range(8))
    style = picker.randrange(3)
    if style == 0:
        return digits
    if style == 1:
        return f"{digits[:4]}-{digits[4:7]}-{digits[7:]}"
    return f"{digits[:4]} {digits[4:7]} {digits[7:]}"


def landline(picker: random.Random) -> str:
    return _seeded(picker).phone_number()


def email(picker: random.Random) -> str:
    return _seeded(picker).free_email()


def personal_url(picker: random.Random) -> str:
    """A URL that points at one person: a profile, a blog, or a personal domain."""
    handle = _seeded(picker).user_name().replace(".", "")
    second = _seeded(picker).user_name().replace(".", "")
    shapes = [
        f"https://{picker.choice(['medium.com/@', 'www.facebook.com/', 'vocus.cc/', 'home.gamer.com.tw/'])}{handle}",
        f"https://www.instagram.com/{handle}/",
        f"https://www.linkedin.com/in/{handle}-{second[:4]}",
        f"https://www.facebook.com/profile.php?id=1000{picker.randint(10**10, 10**11 - 1)}",
        f"https://{handle}.github.io",
        f"https://{handle}.blogspot.com/{picker.randint(2015, 2026)}/{picker.randint(1, 12):02d}/",
        f"https://{handle}-{second}.com",
        f"https://www.{handle}.com.tw/about",
    ]
    return picker.choice(shapes)


def username(picker: random.Random) -> str:
    return _seeded(picker).user_name()


def ig_handle(picker: random.Random) -> str:
    return "@" + username(picker).replace("-", "_")


def password(picker: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%&*"
    return "".join(picker.choice(alphabet) for _ in range(picker.randint(8, 16)))


def api_key(picker: random.Random) -> str:
    prefix = picker.choice(["sk-", "AKIA", "ghp_", "AIza", ""])
    body = "".join(picker.choice(string.ascii_letters + string.digits) for _ in range(picker.randint(24, 40)))
    return prefix + body


def otp(picker: random.Random) -> str:
    return "".join(picker.choice(string.digits) for _ in range(6))


def credit_card(picker: random.Random) -> str:
    number = _seeded(picker).credit_card_number(card_type=picker.choice(["visa", "mastercard", "jcb16"]))
    if picker.random() < 0.5 and len(number) == 16:
        return "-".join(number[i : i + 4] for i in range(0, 16, 4))
    return number


def digits(count: int) -> callable:
    return lambda picker: "".join(picker.choice(string.digits) for _ in range(count))


def public_choice(options: list[str]) -> callable:
    return lambda picker: picker.choice(options)


STORES = [
    "全家便利商店",
    "7-ELEVEN",
    "萊爾富",
    "誠品書店",
    "鼎泰豐",
    "85度C",
    "家樂福",
    "全聯福利中心",
    "路易莎咖啡",
    "小北百貨",
    "寶雅",
    "新光三越",
    "燦坤",
    "屈臣氏",
    "康是美",
    "大創",
]
LANDMARKS = [
    "台北101",
    "故宮博物院",
    "中正紀念堂",
    "西門町",
    "九份老街",
    "駁二藝術特區",
    "日月潭",
    "高雄巨蛋",
    "台中歌劇院",
    "淡水老街",
    "饒河街夜市",
    "逢甲夜市",
    "安平古堡",
    "國父紀念館",
]
LANDMARK_ADDRESSES = [
    "台北市信義區信義路五段7號",
    "台北市士林區至善路二段221號",
    "台北市中正區中山南路21號",
    "高雄市鹽埕區大勇路1號",
    "台中市西屯區惠來路二段101號",
    "台北市信義區仁愛路四段505號",
]
HOTLINES = [
    "1999",
    "110",
    "119",
    "165",
    "1922",
    "1957",
    "113",
    "0800-000-123",
    "(02)2345-6789",
    "4128-000",
    "0800-211-511",
]
ORG_EMAILS = [
    "service@gov.taipei",
    "service@mail.ntpc.gov.tw",
    "cs@pchome.com.tw",
    "service@momoshop.com.tw",
    "contact@ey.gov.tw",
    "press@company.com.tw",
    "info@museum.gov.tw",
]
CELEBRITIES = ["周杰倫", "蔡依林", "五月天", "林書豪", "王建民", "鄧麗君", "張惠妹", "戴資穎", "李安", "侯孝賢"]
PUBLIC_URLS = [
    "https://www.gov.taipei",
    "https://www.cwa.gov.tw",
    "https://www.nhi.gov.tw",
    "https://tixcraft.com",
    "https://www.momoshop.com.tw",
    "https://24h.pchome.com.tw",
]
PRODUCTS = ["藍牙耳機", "行動電源", "保溫瓶", "運動鞋", "電子鍋", "機車安全帽", "鳳梨酥禮盒", "茶葉禮盒", "二手iPhone"]
BANKS = [
    ("台灣銀行", "004"),
    ("土地銀行", "005"),
    ("合作金庫", "006"),
    ("第一銀行", "007"),
    ("華南銀行", "008"),
    ("彰化銀行", "009"),
    ("國泰世華", "013"),
    ("中國信託", "822"),
    ("玉山銀行", "808"),
    ("台新銀行", "812"),
]
HOSPITALS = ["臺大醫院", "榮民總醫院", "長庚紀念醫院", "馬偕紀念醫院", "成大醫院", "中國醫藥大學附設醫院"]
SCHOOLS = ["建國中學", "北一女中", "台中一中", "雄中", "成功國小", "光復國中", "大安國小"]


def bank_name(picker: random.Random) -> str:
    return picker.choice(BANKS)[0]


def bank_code(picker: random.Random) -> str:
    return picker.choice(BANKS)[1]


def time_of_day(picker: random.Random) -> str:
    hour, minute = picker.randint(8, 21), picker.choice([0, 15, 30, 45])
    return f"{hour}:{minute:02d}" if picker.random() < 0.5 else f"{'上午' if hour < 12 else '下午'}{hour % 12 or 12}點"


def amount(picker: random.Random) -> str:
    return f"{picker.randint(1, 500) * 100:,}"


# --- Formats verified by source research on 2026-09-27. Confidence noted per generator. ---

CITY_LETTERS = "ABCDEFGHJKMNPQTUVWXZ"  # household number: city or county letter


def company_id(picker: random.Random) -> str:
    """統一編號: 8 digits. Weights 1,2,1,2,1,2,4,1; each product's digits are summed.
    Valid when the sum is divisible by 5 (relaxed from 10 in 2023). When the 7th digit
    is 7, its product 28 may count as 10 or 1, so sum + 1 divisible by 5 also passes.
    Source: Ministry of Finance notice, verified by research."""
    while True:
        candidate = "".join(picker.choice(string.digits) for _ in range(8))
        if company_id_valid(candidate):
            return candidate


def company_id_valid(number: str) -> bool:
    if len(number) != 8 or not number.isdigit():
        return False
    weights = (1, 2, 1, 2, 1, 2, 4, 1)
    total = sum(sum(divmod(int(digit) * weight, 10)) for digit, weight in zip(number, weights, strict=True))
    if total % 5 == 0:
        return True
    return number[6] == "7" and (total + 1) % 5 == 0


def nhi_card(picker: random.Random) -> str:
    """NHI card number: 12 digits. No structure or checksum is published, so digits are
    uniform. Partly verified: the card data spec also allows letters."""
    number = "0000" + "".join(picker.choice(string.digits) for _ in range(8))
    return number if picker.random() < 0.6 else " ".join(number[i : i + 4] for i in range(0, 12, 4))


def military_id(picker: random.Random) -> str:
    """The service number is the holder's national ID number (Ministry of National Defense).
    tw-PII-bench also writes older numbers as 軍字第 N 號, so a fifth of values use that form."""
    if picker.random() < 0.2:
        return f"軍字第{picker.randint(100000, 9999999)}號"
    return national_id(picker)


def driver_license(picker: random.Random) -> str:
    """The licence number is the holder's national ID number (Wikipedia, no government page found)."""
    return national_id(picker)


def household_no(picker: random.Random) -> str:
    """戶號: city letter, 6 digits, then a check digit (Ministry of the Interior regulation,
    article 3). The check algorithm was not found, so the last digit is random."""
    return picker.choice(CITY_LETTERS) + "".join(picker.choice(string.digits) for _ in range(7))


def medical_license(picker: random.Random) -> str:
    """Physician certificate: 醫字第 + 5 or 6 digits + 號. UNVERIFIED beyond one clinic's site."""
    number = f"{picker.randint(1, 99999):06d}" if picker.random() < 0.8 else f"{picker.randint(1, 99999):05d}"
    wrapper = picker.choice(["醫字第{}號", "醫字第{}號", "醫師字第{}號", "{}"])
    return wrapper.format(number)


def passport(picker: random.Random) -> str:
    """9 digits; current chip passports start with 3 (Wikipedia)."""
    return "3" + "".join(picker.choice(string.digits) for _ in range(8))


def license_plate(picker: random.Random) -> str:
    """Current ABC-1234 (since 2012), and the still-valid AB-1234 and 1234-AB (Wikipedia)."""
    letters = lambda count: "".join(picker.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(count))
    numbers = lambda count: "".join(picker.choice(string.digits) for _ in range(count))
    style = picker.randrange(3)
    if style == 0:
        return f"{letters(3)}-{numbers(4)}"
    return f"{letters(2)}-{numbers(4)}" if style == 1 else f"{numbers(4)}-{letters(2)}"


def bank_account(picker: random.Random) -> str:
    """Account number of 8 to 14 digits; the 3-digit bank code is written separately."""
    number = "".join(picker.choice(string.digits) for _ in range(picker.randint(10, 14)))
    return number if picker.random() < 0.7 else f"{number[:3]}-{number[3:6]}-{number[6:]}"


def line_id(picker: random.Random) -> str:
    """Lowercase letters, digits, '.', '-', '_'. No published length limit; 4 to 20 used here."""
    base = username(picker).lower().replace(" ", "")
    tail = "".join(picker.choice(string.digits + "._") for _ in range(picker.randint(0, 3)))
    return (base + tail)[:20]


def ptt_id(picker: random.Random) -> str:
    """2 to 12 letters or digits, starting with a letter (PTT source code)."""
    first = picker.choice(string.ascii_letters)
    rest = "".join(picker.choice(string.ascii_letters + string.digits) for _ in range(picker.randint(1, 11)))
    return first + rest
