import pytest

from secrag.numbers import check_claim_numbers, extract_numbers, is_grounded, safe_eval

TABLE = "[Amounts in millions]\nNet sales | 391,035 | 383,285\nGross margin percentage | 46.2% | 44.1%"


def ev():
    return extract_numbers(TABLE, skip_years=False)


def test_extract_skips_years_and_noise():
    nums = extract_numbers("In fiscal 2024 (FY2024, Q3, 10-K Item 7) net sales were $391.0 billion [1].")
    assert [n.text for n in nums] == ["$391.0 billion"]
    assert nums[0].value == pytest.approx(391.0e9)


@pytest.mark.parametrize(
    "claim,ok",
    [
        ("net sales were $391,035 million", True),
        ("net sales were $391.0 billion", True),  # rounding of 391,035m
        ("net sales were $391 billion", True),
        ("net sales were $391.1 billion", False),  # wrong digit
        ("net sales were $400.5 billion", False),
        ("gross margin was 46.2%", True),
        ("gross margin was 46%", True),
        ("gross margin was 47.2%", False),
    ],
)
def test_grounding(claim, ok):
    n = extract_numbers(claim)[0]
    assert is_grounded(n, ev()) is ok


def test_calculation_checks():
    good = check_claim_numbers("Net sales grew 2.0% in fiscal 2024.", "(391,035 - 383,285) / 383,285 * 100", ev())
    assert good[0], good[1]
    wrong_result = check_claim_numbers("Net sales grew 3.5% in fiscal 2024.", "(391035-383285)/383285*100", ev())
    assert not wrong_result[0]
    bad_operand = check_claim_numbers("Net sales grew 2.3% in fiscal 2024.", "(392000-383285)/383285*100", ev())
    assert not bad_operand[0] and "operand" in bad_operand[1]
    no_calc = check_claim_numbers("Net sales grew 2.0% in fiscal 2024.", None, ev())
    assert not no_calc[0]  # derived % with no calculation is not grounded


def test_safe_eval_rejects_code():
    assert safe_eval("(10 - 4) / 2") == 3.0
    for bad in ["__import__('os')", "a + 1", "2 ** 10", "[1][0]"]:
        with pytest.raises(Exception):
            safe_eval(bad)
