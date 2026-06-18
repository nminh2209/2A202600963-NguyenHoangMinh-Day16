from src.reflexion_lab.utils import answers_match, normalize_answer

def test_normalize_answer():
    assert normalize_answer("Oxford University!") == "oxford university"

def test_normalize_answer_strips_article():
    assert normalize_answer("An organ") == "organ"

def test_answers_match_short_form():
    assert answers_match("organ", "an organ")
    assert answers_match("Bury St Edmunds", "Bury St Edmunds, Suffolk")
