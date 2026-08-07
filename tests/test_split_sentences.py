from callmind.gateway.session import split_sentences


def test_split_on_sentence_boundary():
    chunks, rest = split_sentences("Hello there. How can I help?")
    assert chunks == ["Hello there. "]
    assert rest == "How can I help?"


def test_no_boundary_keeps_buffer():
    chunks, rest = split_sentences("Hello there")
    assert chunks == []
    assert rest == "Hello there"


def test_long_text_flushes_without_boundary():
    text = "a" * 300
    chunks, rest = split_sentences(text, max_len=180)
    assert chunks == [text]
    assert rest == ""


def test_multiple_sentences():
    chunks, rest = split_sentences("One. Two! Three? Four")
    assert chunks == ["One. ", "Two! ", "Three? "]
    assert rest == "Four"
