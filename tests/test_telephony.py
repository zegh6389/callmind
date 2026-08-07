import base64
import json

from callmind.telephony.base import CallStart, CallStop, MediaChunk
from callmind.telephony.telnyx import TelnyxAdapter
from callmind.telephony.twilio import TwilioAdapter


def _twilio_start_msg():
    return json.dumps({
        "event": "start",
        "streamSid": "MZ123",
        "start": {
            "streamSid": "MZ123",
            "callSid": "CA123",
            "from": "+15551234567",
            "to": "+15559876543",
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    })


def test_twilio_parse_start():
    adapter = TwilioAdapter()
    event = adapter.parse(_twilio_start_msg())
    assert isinstance(event, CallStart)
    assert event.call_id == "CA123"
    assert event.stream_id == "MZ123"
    assert event.from_number == "+15551234567"


def test_twilio_parse_media_inbound():
    adapter = TwilioAdapter()
    adapter.parse(_twilio_start_msg())
    payload = base64.b64encode(b"\xff" * 160).decode()
    msg = json.dumps({
        "event": "media",
        "streamSid": "MZ123",
        "media": {"track": "inbound", "chunk": "7", "payload": payload},
    })
    event = adapter.parse(msg)
    assert isinstance(event, MediaChunk)
    assert event.payload == b"\xff" * 160
    assert event.seq == 7


def test_twilio_parse_media_outbound_ignored():
    adapter = TwilioAdapter()
    msg = json.dumps({
        "event": "media",
        "streamSid": "MZ123",
        "media": {"track": "outbound", "payload": ""},
    })
    assert adapter.parse(msg) is None


def test_twilio_parse_stop():
    adapter = TwilioAdapter()
    event = adapter.parse(json.dumps({"event": "stop", "callSid": "CA123"}))
    assert isinstance(event, CallStop)
    assert event.call_id == "CA123"


def test_twilio_media_message_roundtrip():
    adapter = TwilioAdapter()
    adapter.parse(_twilio_start_msg())
    msg = adapter.media_message(b"\x00" * 160, seq=1)
    assert msg["event"] == "media"
    assert msg["streamSid"] == "MZ123"
    assert base64.b64decode(msg["media"]["payload"]) == b"\x00" * 160


def test_twilio_clear_message():
    adapter = TwilioAdapter()
    assert adapter.clear_message() is None
    adapter.parse(_twilio_start_msg())
    assert adapter.clear_message() == {"event": "clear", "streamSid": "MZ123"}


def test_twilio_media_malformed_base64_does_not_crash():
    adapter = TwilioAdapter()
    adapter.parse(_twilio_start_msg())
    msg = json.dumps({
        "event": "media",
        "streamSid": "MZ123",
        "media": {"track": "inbound", "chunk": "8", "payload": "!!!not-base64!!!"},
    })
    assert adapter.parse(msg) is None


def test_twilio_media_empty_payload_skipped():
    adapter = TwilioAdapter()
    adapter.parse(_twilio_start_msg())
    msg = json.dumps({
        "event": "media",
        "streamSid": "MZ123",
        "media": {"track": "inbound", "chunk": "9", "payload": ""},
    })
    assert adapter.parse(msg) is None


def _telnyx_start_msg():
    return json.dumps({
        "event": "start",
        "sequence_number": "1",
        "start": {
            "call_control_id": "v2:CC123",
            "call_session_id": "cs-1",
            "from": "+15551234567",
            "to": "+14375249536",
            "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
        },
        "stream_id": "ST-9",
    })


def test_telnyx_parse_start():
    adapter = TelnyxAdapter()
    event = adapter.parse(_telnyx_start_msg())
    assert isinstance(event, CallStart)
    assert event.call_id == "v2:CC123"
    assert event.stream_id == "ST-9"
    assert event.to_number == "+14375249536"


def test_telnyx_parse_media():
    adapter = TelnyxAdapter()
    adapter.parse(_telnyx_start_msg())
    payload = base64.b64encode(b"\xff" * 160).decode()
    msg = json.dumps({
        "event": "media",
        "sequence_number": "4",
        "media": {"track": "inbound", "chunk": "2", "timestamp": "5", "payload": payload},
        "stream_id": "ST-9",
    })
    event = adapter.parse(msg)
    assert isinstance(event, MediaChunk)
    assert event.payload == b"\xff" * 160
    assert event.seq == 2


def test_telnyx_parse_stop():
    adapter = TelnyxAdapter()
    event = adapter.parse(json.dumps({
        "event": "stop",
        "stop": {"call_control_id": "v2:CC123"},
        "stream_id": "ST-9",
    }))
    assert isinstance(event, CallStop)
    assert event.call_id == "v2:CC123"


def test_telnyx_connected_and_dtmf_ignored():
    adapter = TelnyxAdapter()
    assert adapter.parse(json.dumps({"event": "connected", "version": "1.0.0"})) is None
    assert adapter.parse(json.dumps({"event": "dtmf", "dtmf": {"digit": "1"}})) is None


def test_telnyx_media_message_and_clear():
    adapter = TelnyxAdapter()
    msg = adapter.media_message(b"\x00" * 160, seq=1)
    assert msg == {"event": "media", "media": {"payload": base64.b64encode(b"\x00" * 160).decode()}}
    assert adapter.clear_message() == {"event": "clear"}


def test_telnyx_media_malformed_base64_does_not_crash():
    adapter = TelnyxAdapter()
    adapter.parse(_telnyx_start_msg())
    msg = json.dumps({
        "event": "media",
        "sequence_number": "5",
        "media": {"track": "inbound", "payload": "@@@not-base64@@@"},
        "stream_id": "ST-9",
    })
    assert adapter.parse(msg) is None
    assert adapter.parse(json.dumps({"event": "media", "media": {"track": "inbound"}})) is None
