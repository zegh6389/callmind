from datetime import UTC

import pytest

from callmind.tools.account import AccountTool
from callmind.tools.booking import BookingTool
from callmind.tools.router import ToolRouter


@pytest.fixture
def booking():
    return BookingTool()


@pytest.fixture
def account():
    return AccountTool()


@pytest.fixture
def router():
    return ToolRouter()


def test_booking_valid(booking):
    res = await_(
        booking.run,
        {
            "title": "Doctor visit",
            "date": "2026-08-15",
            "time": "14:00",
            "caller_name": "Alice",
        },
    )
    assert res.success is True
    assert "Doctor visit" in res.summary
    assert "2026-08-15" in res.summary
    assert res.data["event_id"]


def test_booking_missing_required_field(booking):
    res = await_(
        booking.run,
        {"title": "Visit", "date": "2026-08-15"},
    )
    assert res.success is False
    assert "time" in res.error.lower() or "missing" in res.error.lower()


def test_booking_invalid_date_format(booking):
    res = await_(
        booking.run,
        {"title": "Visit", "date": "tomorrow", "time": "14:00"},
    )
    assert res.success is False


def test_account_known_phone(account):
    res = await_(
        account.run,
        {"caller_phone": "+15551111111"},
    )
    assert res.success is True
    assert res.data.get("account_id") or res.data.get("status")


def test_account_missing_phone(account):
    res = await_(account.run, {})
    assert res.success is False


def test_router_dispatch_booking(router):
    res = await_(
        router.dispatch,
        "booking",
        {"title": "X", "date": "2026-08-15", "time": "14:00", "caller_name": "A"},
        call_id="c1",
        business_id="b1",
    )
    assert res.success is True


def test_router_dispatch_unknown_intent(router):
    res = await_(router.dispatch, "smalltalk", {}, call_id="c1", business_id="b1")
    assert res is None


def test_router_extract_booking_params(router):
    from datetime import datetime, timedelta

    params = router.extract_params("booking", "Can I book a slot tomorrow at 2pm?")
    assert params.get("date") == (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    assert params.get("time") == "14:00"


def test_router_extract_account_params_phone(router):
    params = router.extract_params("account_status", "what's the status on 555-123-4567?")
    assert "5551234567" in params.get("caller_phone", "")


def test_router_phone_not_lucky_number(router):
    # 8-9 digit order/ticket ids are not phones; must not hallucinate an account.
    params = router.extract_params("account_status", "My order number is 12345678, when will it arrive?")
    assert not params.get("caller_phone")
    params = router.extract_params("account_status", "Use ticket 123456789 please")
    assert not params.get("caller_phone")


def test_router_phone_accepts_plus_country(router):
    params = router.extract_params("account_status", "that's +1 416 555-0199")
    assert params.get("caller_phone") == "14165550199"


def test_router_extract_returns_empty_for_unknown(router):
    params = router.extract_params("smalltalk", "hi there")
    assert params == {}


def test_router_available_tools(router):
    names = router.available_tools()
    assert "booking" in names
    assert "account_status" in names


# --- async helper: tests are sync, tools are async ---
import asyncio


def await_(fn, *args, **kwargs):
    kw = {"call_id": "test-call", "business_id": "test-biz", **kwargs}
    return asyncio.run(fn(*args, **kw))