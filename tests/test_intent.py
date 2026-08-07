
from callmind.brain.intent import _extract_json


def test_extract_json_plain():
    assert _extract_json('{"intent": "faq", "confidence": 0.9}') == {"intent": "faq", "confidence": 0.9}


def test_extract_json_fenced():
    text = '```json\n{"intent":"booking","confidence":0.7}\n```'
    assert _extract_json(text) == {"intent": "booking", "confidence": 0.7}


def test_extract_json_embedded():
    text = 'sure, here you go: {"intent":"smalltalk","confidence":0.4}'
    assert _extract_json(text) == {"intent": "smalltalk", "confidence": 0.4}


def test_extract_json_garbage():
    assert _extract_json("not json at all") is None


def test_extract_json_truncated():
    text = '{"intent": "faq", "confidence": 0.'
    assert _extract_json(text) is None