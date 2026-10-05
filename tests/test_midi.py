"""Unit tests for the strict SMF parser and timeline normalizer."""

import unittest
from fractions import Fraction

from app.midi import (
    MAX_TRACKS_PLUS_EVENTS,
    MidiError,
    build_tempo_segments,
    normalize,
    parse,
    smpte_time_at_tick,
    time_at_tick,
    TempoEvent,
)


# -- SMF builders ---------------------------------------------------------


def varlen(n):
    out = bytes([n & 0x7F])
    n >>= 7
    while n:
        out = bytes([(n & 0x7F) | 0x80]) + out
        n >>= 7
    return out


def track(payload):
    return b"MTrk" + len(payload).to_bytes(4, "big") + payload


def header(fmt, ntracks, division):
    return (
        b"MThd"
        + (6).to_bytes(4, "big")
        + fmt.to_bytes(2, "big")
        + ntracks.to_bytes(2, "big")
        + division.to_bytes(2, "big")
    )


def tempo(delta, us):
    return varlen(delta) + bytes([0xFF, 0x51, 0x03]) + us.to_bytes(3, "big")


def ev(delta, *bs):
    return varlen(delta) + bytes(bs)


EOT = ev(0, 0xFF, 0x2F, 0x00)


def fractions(result):
    return [Fraction(e["time_us"]["numerator"], e["time_us"]["denominator"])
            for e in result["events"]]


# -- header / chunk validation --------------------------------------------


class HeaderTests(unittest.TestCase):
    def test_bad_magic(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"NOPE" + bytes(10))
        self.assertEqual(ctx.exception.code, "bad_magic")
        self.assertEqual(ctx.exception.offset, 0)

    def test_truncated_header(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"MThd\x00\x00")
        self.assertEqual(ctx.exception.code, "truncated_header")
        self.assertEqual(ctx.exception.offset, 6)

    def test_bad_header_length(self):
        data = b"MThd" + (8).to_bytes(4, "big") + bytes(8)
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "bad_header_length")
        self.assertEqual(ctx.exception.offset, 4)

    def test_format_2_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(2, 2, 480))
        self.assertEqual(ctx.exception.code, "unsupported_format")
        self.assertEqual(ctx.exception.offset, 8)

    def test_invalid_frame_rate_code_rejected(self):
        # -26, -28, -2 and -128 are all outside {-24, -25, -29, -30}.
        for word in (0xE628, 0xE428, 0xFE28, 0x8028):
            with self.assertRaises(MidiError) as ctx:
                parse(header(1, 1, word))
            self.assertEqual(ctx.exception.code, "invalid_frame_rate", word)
            self.assertEqual(ctx.exception.offset, 12, word)

    def test_zero_ticks_per_frame_rejected(self):
        for word in (0xE800, 0xE700, 0xE300, 0xE200):
            with self.assertRaises(MidiError) as ctx:
                parse(header(1, 1, word))
            self.assertEqual(ctx.exception.code, "invalid_ticks_per_frame", word)
            self.assertEqual(ctx.exception.offset, 13, word)

    def test_smpte_accepted_codes(self):
        for word, fps in (
            (0xE828, Fraction(24, 1)),
            (0xE728, Fraction(25, 1)),
            (0xE328, Fraction(30000, 1001)),
            (0xE228, Fraction(30, 1)),
        ):
            data = header(1, 1, word) + track(ev(0, 0x90, 60, 100) + EOT)
            parsed = parse(data)
            self.assertTrue(parsed.is_smpte)
            self.assertEqual(parsed.fps, fps)
            self.assertEqual(parsed.ticks_per_frame, 40)
            # Tempo meta events remain legal but build no tempo map.
            self.assertEqual(parsed.tempo_events, [])

    def test_zero_ppqn_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 0))
        self.assertEqual(ctx.exception.code, "invalid_division")

    def test_zero_tracks_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 0, 480))
        self.assertEqual(ctx.exception.code, "no_tracks")

    def test_format_0_requires_single_track(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(0, 2, 480))
        self.assertEqual(ctx.exception.code, "bad_track_count")

    def test_missing_track_chunk(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480))
        self.assertEqual(ctx.exception.code, "truncated_track_header")
        self.assertEqual(ctx.exception.offset, 14)

    def test_wrong_chunk_type(self):
        data = header(1, 1, 480) + b"LIST" + bytes(4)
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "bad_chunk")
        self.assertEqual(ctx.exception.offset, 14)

    def test_truncated_track_body(self):
        data = header(1, 1, 480) + b"MTrk" + (100).to_bytes(4, "big") + b"\x00\xff"
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "truncated_track")
        self.assertEqual(ctx.exception.offset, 18)

    def test_trailing_bytes_rejected(self):
        data = header(1, 1, 480) + track(EOT) + b"\x00"
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "trailing_bytes")
        self.assertEqual(ctx.exception.offset, len(data) - 1)

    def test_file_too_large(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"\x00" * (1 << 20 | 1))
        self.assertEqual(ctx.exception.code, "file_too_large")


# -- event-level validation ------------------------------------------------


class EventTests(unittest.TestCase):
    def test_varlen_too_long(self):
        payload = b"\x80\x80\x80\x80\x00" + ev(0, 0x90, 60, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "varlen_too_long")
        self.assertEqual(ctx.exception.offset, 22)

    def test_unterminated_varlen(self):
        payload = b"\x80\x80"
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_varlen")
        self.assertEqual(ctx.exception.offset, 22)

    def test_missing_status_byte(self):
        payload = ev(0, 0x90, 60, 100) + varlen(10)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_orphan_running_status(self):
        payload = ev(0, 60, 100)  # data bytes before any status
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")
        self.assertEqual(ctx.exception.offset, 23)

    def test_running_status_cancelled_by_meta(self):
        payload = ev(0, 0x90, 60, 100) + ev(0, 0xFF, 0x01, 0x00) + ev(0, 62, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")

    def test_running_status_cancelled_by_sysex(self):
        payload = ev(0, 0x90, 60, 100) + ev(0, 0xF0, 0x00) + ev(0, 62, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")

    def test_running_status_ok(self):
        payload = ev(0, 0x90, 60, 100) + ev(10, 62, 100) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 2)
        self.assertEqual(result["events"][1]["data"], [62, 100])
        self.assertEqual(result["events"][1]["tick"], 10)

    def test_illegal_status(self):
        payload = ev(0, 0xF8)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "illegal_status")
        self.assertEqual(ctx.exception.offset, 23)

    def test_data_byte_high_bit(self):
        payload = ev(0, 0x90, 60, 0x80)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "invalid_data_byte")
        self.assertEqual(ctx.exception.offset, 25)

    def test_truncated_channel_event(self):
        payload = ev(0, 0x90, 60)  # note on needs two data bytes
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_program_change_single_data_byte(self):
        payload = ev(0, 0xC3, 5) + ev(0, 0xD0, 7) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(
            [e["type"] for e in result["events"]],
            ["program_change", "channel_pressure"],
        )
        self.assertEqual(result["events"][0]["channel"], 3)

    def test_truncated_meta_event(self):
        payload = ev(0, 0xFF, 0x01, 0x05) + b"ab"  # declares 5, gives 2
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_bad_tempo_length(self):
        payload = ev(0, 0xFF, 0x51, 0x02, 0x07, 0xA1)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "bad_meta_length")

    def test_zero_tempo_rejected(self):
        payload = tempo(0, 0)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "invalid_tempo")

    def test_bad_end_of_track_length(self):
        payload = ev(0, 0xFF, 0x2F, 0x01, 0x00)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "bad_meta_length")

    def test_truncated_sysex(self):
        payload = ev(0, 0xF0, 0x05) + b"ab"
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_sysex_and_meta_accepted(self):
        payload = (
            ev(0, 0xF0, 0x03, 0x7E, 0x7F, 0x09)
            + ev(0, 0xFF, 0x03, 0x04) + b"name"
            + ev(0, 0x90, 60, 100)
            + EOT
        )
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 1)


# -- limits ------------------------------------------------------------------


class LimitTests(unittest.TestCase):
    def test_track_and_event_total_limit(self):
        # 1 track + 9999 channel events == 10000: accepted.
        payload = ev(0, 0x90, 60, 100) + ev(0, 60, 100) * 9998 + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 9999)

        # One more channel event crosses the limit.
        payload = ev(0, 0x90, 60, 100) + ev(0, 60, 100) * 9999 + EOT
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "limit_exceeded")

    def test_track_count_limit(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, MAX_TRACKS_PLUS_EVENTS + 1, 480))
        self.assertEqual(ctx.exception.code, "limit_exceeded")
        self.assertEqual(ctx.exception.offset, 10)


# -- tempo map and timing ----------------------------------------------------


class TempoTests(unittest.TestCase):
    def test_default_tempo(self):
        payload = ev(0, 0x90, 60, 100) + ev(480, 0x80, 60, 0) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(fractions(result), [Fraction(0), Fraction(500000)])

    def test_reduced_fraction(self):
        payload = ev(1, 0x90, 60, 100) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(fractions(result), [Fraction(3125, 3)])
        self.assertEqual(result["events"][0]["time_us"]["fraction"], "3125/3")

    def test_tempo_takes_effect_at_its_tick(self):
        conductor = tempo(0, 500000) + tempo(480, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + ev(480, 0x80, 60, 0) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        # Tick 480 itself is still priced with the old tempo; the new tempo
        # only governs the interval that starts at tick 480.
        self.assertEqual(
            fractions(result), [Fraction(500000), Fraction(750000)]
        )

    def test_same_tick_last_tempo_wins(self):
        conductor = tempo(0, 600000) + tempo(0, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(250000)])

    def test_format_1_ignores_non_first_track_tempos(self):
        conductor = tempo(0, 500000) + EOT
        notes = tempo(0, 1000) + ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(500000)])

    def test_format_0_uses_own_tempos(self):
        payload = tempo(0, 250000) + ev(480, 0x90, 60, 100) + EOT
        result = normalize(header(0, 1, 480) + track(payload))
        self.assertEqual(result["format"], 0)
        self.assertEqual(fractions(result), [Fraction(250000)])

    def test_multi_segment_tempo_map(self):
        segments = build_tempo_segments(
            [TempoEvent(0, 500000), TempoEvent(480, 250000), TempoEvent(960, 1000000)]
        )
        self.assertEqual(time_at_tick(0, segments, 480), Fraction(0))
        self.assertEqual(time_at_tick(240, segments, 480), Fraction(250000))
        self.assertEqual(time_at_tick(480, segments, 480), Fraction(500000))
        self.assertEqual(time_at_tick(960, segments, 480), Fraction(750000))
        self.assertEqual(time_at_tick(1200, segments, 480), Fraction(1250000))

    def test_tempo_beyond_last_event_is_harmless(self):
        conductor = tempo(0, 500000) + tempo(9999, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(500000)])


# -- ordering ----------------------------------------------------------------


class OrderingTests(unittest.TestCase):
    def test_stable_order_by_tick_track_order(self):
        t0 = ev(240, 0x90, 60, 100) + EOT
        t1 = ev(0, 0x91, 61, 100) + ev(240, 0x81, 61, 0) + EOT
        t2 = ev(120, 0x92, 62, 100) + ev(120, 0x82, 62, 0) + EOT
        data = header(1, 3, 480) + track(t0) + track(t1) + track(t2)
        result = normalize(data)
        keys = [(e["tick"], e["track"], e["order"]) for e in result["events"]]
        self.assertEqual(
            keys,
            [(0, 1, 0), (120, 2, 0), (240, 0, 0), (240, 1, 1), (240, 2, 1)],
        )
        self.assertEqual(
            fractions(result),
            [Fraction(0)] + [Fraction(125000)] + [Fraction(250000)] * 3,
        )


# -- SMPTE timing ------------------------------------------------------------


class SmpteTests(unittest.TestCase):
    def test_time_formula_directly(self):
        self.assertEqual(
            smpte_time_at_tick(0, Fraction(25, 1), 40), Fraction(0)
        )
        # 25 fps * 40 ticks/frame = 1000 ticks/s, so 1 tick = 1000 us.
        self.assertEqual(
            smpte_time_at_tick(1, Fraction(25, 1), 40), Fraction(1000)
        )
        # 24 fps, 40 ticks/frame: one frame is 1/24 s = 125000/3 us.
        self.assertEqual(
            smpte_time_at_tick(40, Fraction(24, 1), 40), Fraction(125000, 3)
        )
        # 30 fps, 100 ticks/frame: 3000 ticks/s.
        self.assertEqual(
            smpte_time_at_tick(3000, Fraction(30, 1), 100), Fraction(1_000_000)
        )
        # 30000/1001 fps: one frame = 100100/3 us.
        self.assertEqual(
            smpte_time_at_tick(40, Fraction(30000, 1001), 40),
            Fraction(100100, 3),
        )

    def test_integer_frame_rate_timeline(self):
        # -25 fps, 40 ticks/frame -> 1000 ticks/second -> 1000 us/tick.
        payload = (
            ev(0, 0x90, 60, 100)       # tick 0
            + ev(100, 0x80, 60, 0)     # tick 100 -> 100000 us
            + EOT
        )
        result = normalize(header(1, 1, 0xE728) + track(payload))
        self.assertEqual(fractions(result), [Fraction(0), Fraction(100000)])

    def test_24fps_reduced_fraction(self):
        payload = ev(1, 0x90, 60, 100) + EOT  # 1 tick at 24fps*40
        result = normalize(header(0, 1, 0xE828) + track(payload))
        # 1_000_000 / 960 = 3125/3 us
        self.assertEqual(fractions(result), [Fraction(3125, 3)])
        self.assertEqual(
            result["events"][0]["time_us"]["fraction"], "3125/3"
        )

    def test_2997_drop_frame_rate(self):
        # -29 means 30000/1001 fps, 40 ticks/frame.
        payload = (
            ev(40, 0x90, 60, 100)     # tick 40 = one frame
            + ev(40, 0x80, 60, 0)     # tick 80 = two frames
            + EOT
        )
        result = normalize(header(1, 1, 0xE328) + track(payload))
        self.assertEqual(
            fractions(result),
            [Fraction(100100, 3), Fraction(200200, 3)],
        )

    def test_30fps(self):
        payload = ev(1200, 0x90, 60, 100) + EOT  # 30fps*40 = 1200 t/s
        result = normalize(header(1, 1, 0xE228) + track(payload))
        self.assertEqual(fractions(result), [Fraction(1_000_000)])

    def test_tempo_events_do_not_change_smpte_time(self):
        # Even an absurd, but legal, tempo on track 0 must not move events.
        conductor = (
            tempo(0, 1)
            + tempo(20, 12_000_000)
            + EOT
        )
        notes = (
            ev(100, 0x90, 60, 100)
            + ev(100, 0x80, 60, 0)
            + EOT
        )
        data = header(1, 2, 0xE728) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(100000), Fraction(200000)])

    def test_illegal_tempo_still_rejected_in_smpte_file(self):
        payload = tempo(0, 0) + EOT
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 0xE728) + track(payload))
        self.assertEqual(ctx.exception.code, "invalid_tempo")

    def test_multitrack_stable_ordering(self):
        t0 = ev(50, 0x90, 60, 100) + EOT
        t1 = ev(0, 0x91, 61, 100) + ev(50, 0x81, 61, 0) + EOT
        data = header(1, 2, 0xE728) + track(t0) + track(t1)
        result = normalize(data)
        keys = [(e["tick"], e["track"], e["order"]) for e in result["events"]]
        self.assertEqual(keys, [(0, 1, 0), (50, 0, 0), (50, 1, 1)])
        self.assertEqual(
            fractions(result),
            [Fraction(0), Fraction(50000), Fraction(50000)],
        )

    def test_response_shape_smpte(self):
        payload = ev(0, 0x90, 60, 100) + EOT
        result = normalize(header(1, 1, 0xE328) + track(payload))
        self.assertNotIn("ppqn", result)
        td = result["time_division"]
        self.assertEqual(td["kind"], "smpte")
        self.assertEqual(td["ticks_per_frame"], 40)
        self.assertEqual(
            td["frame_rate"],
            {
                "numerator": 30000,
                "denominator": 1001,
                "fraction": "30000/1001",
            },
        )

    def test_response_shape_integer_frame_rate(self):
        payload = ev(0, 0x90, 60, 100) + EOT
        result = normalize(header(0, 1, 0xE801) + track(payload))  # 24fps, 1 tpf
        self.assertNotIn("ppqn", result)
        td = result["time_division"]
        self.assertEqual(td["kind"], "smpte")
        self.assertEqual(td["ticks_per_frame"], 1)
        self.assertEqual(td["frame_rate"]["fraction"], "24/1")

    def test_ppqn_response_unchanged(self):
        payload = ev(0, 0x90, 60, 100) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["ppqn"], 480)
        self.assertNotIn("time_division", result)
        self.assertEqual(set(result["events"][0]), {
            "tick", "track", "order", "type", "channel", "data", "time_us"
        })

    def test_structural_error_inside_smpte_track_has_offset(self):
        data = header(1, 1, 0xE728) + track(ev(0, 0xF8))
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "illegal_status")
        self.assertEqual(ctx.exception.offset, 23)

    def test_truncated_smpte_file_has_offset(self):
        data = header(1, 1, 0xE728) + b"MTrk" + (100).to_bytes(4, "big") + b"\x00"
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "truncated_track")
        self.assertEqual(ctx.exception.offset, 18)


if __name__ == "__main__":
    unittest.main()
