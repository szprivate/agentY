"""How long an API key has been sitting in .env, and when to say so.

The point of the ledger is that rotating a key is the ONLY thing that has to
happen for the warning to stop. Anything that made someone acknowledge a reminder
instead would be a reminder people dismiss, so most of what is checked here is
that the clock restarts by itself and that nothing else restarts it.
"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils import key_age

NOW = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)


def ago(days):
    return (NOW - timedelta(days=days)).isoformat(timespec="seconds")


class WhichKeysAreCredentials(unittest.TestCase):
    """Narrower than "does the name look secret-ish", on purpose."""

    def test_the_credentials(self):
        for name in ("HF_TOKEN", "ANTHROPIC_API_KEY", "SLACK_BOT_TOKEN",
                     "SOME_SECRET", "DB_PASSWORD", "SECRET_VALUE",
                     "MY_KEY_FOR_THING"):
            self.assertTrue(key_age.is_secret_key(name), name)

    def test_it_agrees_with_the_panel_about_what_is_a_secret(self):
        """agent_settings.js renders a password field for /KEY|TOKEN|SECRET|PASSWORD/i.

        If the two disagreed, a name this called ordinary and the panel called
        secret would be shown as a password field and sent to the browser in
        plaintext — masked to the eye and not to the wire.
        """
        import re
        panel = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
        for name in ("HF_TOKEN", "ANTHROPIC_API_KEY", "SECRET_VALUE",
                     "DB_PASSWORD", "COMFYUI_API_KEY", "SOME_KEYSTORE"):
            with self.subTest(name=name):
                self.assertEqual(bool(panel.search(name)),
                                 key_age.is_secret_key(name), name)

    def test_the_settings_that_live_beside_them(self):
        """A rotation reminder about a URL teaches people to ignore reminders.

        DASHSCOPE_BASE_URL and SLACK_ALLOWED_USERS are in the same file and are
        not secrets; the settings UI's own isSecret() would flag neither, and this
        must agree with it or the two disagree about what is masked.
        """
        for name in ("DASHSCOPE_BASE_URL", "SLACK_ALLOWED_USERS", "COMFYUI_URL", ""):
            self.assertFalse(key_age.is_secret_key(name), name)


class TheLedger(unittest.TestCase):
    def test_it_never_stores_the_key(self):
        """The thing warning about credentials must not become the leak.

        A file that recorded the keys in order to nag about the keys would be
        strictly worse than not having it.
        """
        ledger = key_age.record({"HF_TOKEN": "hf_secret_value"}, {}, now=NOW)
        serialised = json.dumps(ledger)
        self.assertNotIn("hf_secret_value", serialised)
        self.assertIn("fp", ledger["HF_TOKEN"])

    def test_an_unchanged_key_keeps_its_original_date(self):
        first = key_age.record({"HF_TOKEN": "abc"}, {}, now=NOW - timedelta(days=100))
        again = key_age.record({"HF_TOKEN": "abc"}, first, now=NOW)
        self.assertEqual(first["HF_TOKEN"]["first_seen"],
                         again["HF_TOKEN"]["first_seen"])

    def test_rotating_restarts_the_clock_with_no_other_action(self):
        old = {"HF_TOKEN": {"fp": key_age.fingerprint("abc"), "first_seen": ago(400)}}
        new = key_age.record({"HF_TOKEN": "xyz"}, old, now=NOW)
        self.assertEqual(new["HF_TOKEN"]["first_seen"], NOW.isoformat(timespec="seconds"))
        self.assertEqual(key_age.stale({"HF_TOKEN": "xyz"}, new, 30, NOW), [])

    def test_a_rotation_we_watched_is_not_an_estimate(self):
        old = {"HF_TOKEN": {"fp": key_age.fingerprint("abc"),
                            "first_seen": ago(400), "estimated": True}}
        new = key_age.record({"HF_TOKEN": "xyz"}, old, now=NOW)
        self.assertNotIn("estimated", new["HF_TOKEN"])

    def test_clearing_a_key_drops_it(self):
        old = {"HF_TOKEN": {"fp": key_age.fingerprint("abc"), "first_seen": ago(400)}}
        self.assertEqual(key_age.record({"HF_TOKEN": ""}, old, now=NOW), {})
        self.assertEqual(key_age.record({}, old, now=NOW), {})

    def test_a_new_install_is_seeded_from_the_env_files_timestamp(self):
        """There is no way to learn when an existing key was issued.

        .env's mtime is the honest lower bound, and it is never LATER than seeding
        from "now" — so a checkout whose .env was last touched two years ago warns
        on the first run instead of in thirty days.
        """
        with TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("HF_TOKEN=abc\n", encoding="utf-8")
            import os
            old = (NOW - timedelta(days=400)).timestamp()
            os.utime(env, (old, old))
            ledger = key_age.record({"HF_TOKEN": "abc"}, {}, env_path=env, now=NOW)
            self.assertTrue(ledger["HF_TOKEN"]["estimated"])
            self.assertGreater(key_age.age_days(ledger["HF_TOKEN"]["first_seen"], NOW), 399)

    def test_without_a_file_to_read_it_seeds_from_now(self):
        ledger = key_age.record({"HF_TOKEN": "abc"}, {}, env_path=None, now=NOW)
        self.assertNotIn("estimated", ledger["HF_TOKEN"])
        self.assertEqual(ledger["HF_TOKEN"]["first_seen"], NOW.isoformat(timespec="seconds"))


class TheReport(unittest.TestCase):
    def setUp(self):
        self.env = {"HF_TOKEN": "a", "ANTHROPIC_API_KEY": "b", "OPENAI_API_KEY": "c"}
        self.ledger = {
            "HF_TOKEN": {"fp": key_age.fingerprint("a"), "first_seen": ago(400)},
            "ANTHROPIC_API_KEY": {"fp": key_age.fingerprint("b"), "first_seen": ago(31)},
            "OPENAI_API_KEY": {"fp": key_age.fingerprint("c"), "first_seen": ago(2)},
        }

    def test_oldest_first(self):
        names = [e["key"] for e in key_age.report(self.env, self.ledger, 30, NOW)]
        self.assertEqual(names, ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"])

    def test_the_boundary_is_inclusive(self):
        # 30 days old with a 30-day limit warns. A key that is exactly at the
        # limit is the one the reminder is for.
        ledger = {"K_TOKEN": {"fp": key_age.fingerprint("x"), "first_seen": ago(30)}}
        self.assertTrue(key_age.report({"K_TOKEN": "x"}, ledger, 30, NOW)[0]["stale"])

    def test_zero_turns_it_off(self):
        self.assertEqual(key_age.stale(self.env, self.ledger, 0, NOW), [])
        self.assertFalse(any(e["stale"] for e in key_age.report(self.env, self.ledger, 0, NOW)))

    def test_a_key_removed_from_env_is_not_reported(self):
        report = key_age.report({"HF_TOKEN": "a"}, self.ledger, 30, NOW)
        self.assertEqual([e["key"] for e in report], ["HF_TOKEN"])

    def test_an_unparseable_date_is_skipped_rather_than_crashing(self):
        # The ledger is hand-editable and survives version changes; a bad line costs
        # one key's warning, never a startup.
        ledger = {"HF_TOKEN": {"fp": "x", "first_seen": "not a date"}}
        self.assertEqual(key_age.report({"HF_TOKEN": "a"}, ledger, 30, NOW), [])

    def test_a_naive_timestamp_is_read_as_utc(self):
        ledger = {"HF_TOKEN": {"fp": "x", "first_seen": "2025-01-01T00:00:00"}}
        report = key_age.report({"HF_TOKEN": "a"}, ledger, 30, NOW)
        self.assertTrue(report[0]["stale"])


class TheWarning(unittest.TestCase):
    def test_nothing_stale_says_nothing(self):
        self.assertEqual(key_age.warning_lines([], 30), [])

    def test_it_says_how_to_make_it_stop(self):
        """A warning that cannot be acted on gets switched off rather than fixed.

        The action here IS the thing we want to happen, so it has to be in the
        message — along with the setting, because someone who disagrees should
        turn it down deliberately rather than learn to skip it.
        """
        lines = key_age.warning_lines(
            [{"key": "HF_TOKEN", "age_days": 400.0, "estimated": False}], 30)
        joined = " ".join(lines)
        self.assertIn("HF_TOKEN", joined)
        self.assertIn("Rotate", joined)
        self.assertIn("api_key_max_age_days", joined)

    def test_an_estimate_is_marked_as_one(self):
        lines = key_age.warning_lines(
            [{"key": "HF_TOKEN", "age_days": 400.0, "estimated": True}], 30)
        self.assertIn("at least", " ".join(lines))


class SavingAndLoading(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "key_ages.json"
            ledger = key_age.record({"HF_TOKEN": "abc"}, {}, now=NOW)
            self.assertTrue(key_age.save(path, ledger))
            self.assertEqual(key_age.load(path), ledger)
            self.assertEqual(path.stat().st_mode & 0o077, 0,
                             "the ledger names which credentials exist")

    def test_a_corrupt_ledger_is_an_empty_one(self):
        """Losing it costs a reset clock. Nothing here is worth failing a start."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "key_ages.json"
            path.write_text("{ not json", encoding="utf-8")
            self.assertEqual(key_age.load(path), {})

    def test_a_missing_ledger_is_an_empty_one(self):
        self.assertEqual(key_age.load(Path("/nonexistent/key_ages.json")), {})


if __name__ == "__main__":
    unittest.main()
