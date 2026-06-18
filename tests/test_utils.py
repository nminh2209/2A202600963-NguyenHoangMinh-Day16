from src.reflexion_lab.utils import answers_match, normalize_answer

def test_normalize_answer():
    assert normalize_answer("Oxford University!") == "oxford university"

def test_normalize_answer_strips_article():
    assert normalize_answer("An organ") == "organ"

def test_answers_match_programming_language():
    assert answers_match("C programming language", "C")

def test_polish_answer_programming_language():
    from src.reflexion_lab.utils import polish_answer
    assert polish_answer("What language?", "C programming language") == "C"

def test_polish_answer_no_academy_award():
    from src.reflexion_lab.utils import polish_answer
    q = "What award did the director receive from the Academy?"
    assert polish_answer(q, "No award") == "no Academy Award win"

def test_answers_match_short_form():
    assert answers_match("organ", "an organ")
    assert answers_match("Bury St Edmunds", "Bury St Edmunds, Suffolk")
