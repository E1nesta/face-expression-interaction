from input.text_input import AsyncTextInput


def test_submit_text_updates_latest_state():
    reader = AsyncTextInput(clock=lambda: 10.0)

    reader.submit_text("  我今天有点累  ")

    state = reader.latest_state()
    assert state.text == "  我今天有点累  "
    assert state.normalized_text == "我今天有点累"
    assert state.timestamp == 10.0


def test_empty_text_is_kept_as_empty_state():
    reader = AsyncTextInput(clock=lambda: 12.0)

    reader.submit_text("   ")

    state = reader.latest_state()
    assert state.text == "   "
    assert state.normalized_text == ""
    assert state.timestamp == 12.0


def test_background_input_reads_until_eof():
    values = iter(["开心", "困"])

    def fake_input():
        try:
            return next(values)
        except StopIteration as exc:
            raise EOFError from exc

    reader = AsyncTextInput(input_func=fake_input, clock=lambda: 20.0)

    reader.start()
    reader.join(timeout=1.0)

    assert reader.latest_state().text == "困"
