#!/usr/bin/env python3
"""Unit tests of the runner's pure parts (no AWS call): the session archive extraction, the session-start retry, the summary of lost trials.
Usage: .venv/bin/python scripts/test_runner_v2.py"""
import io, json, os, pathlib, sys, tarfile, tempfile, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p2_v2 as p2


def make_tar(members):
    """members: list of (name, kind, data-or-linkname). Returns the gz tar bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, kind, v in members:
            ti = tarfile.TarInfo(name)
            if kind == "file":
                ti.size = len(v)
                tf.addfile(ti, io.BytesIO(v))
            elif kind == "symlink":
                ti.type, ti.linkname = tarfile.SYMTYPE, v
                tf.addfile(ti)
            elif kind == "device":
                ti.type = tarfile.CHRTYPE
                tf.addfile(ti)
    return buf.getvalue()


class Extract(unittest.TestCase):
    def test_bad_members_are_skipped_and_reward_survives(self):
        reward = json.dumps({"reward": 1.0}).encode()
        raw = make_tar([
            ("agent/codex-home/tmp/arg0/codex-arg0X/applypatch", "symlink", "/usr/local/bin/codex"),  # absolute symlink: the real failure
            ("agent/escape-link", "symlink", "../../outside"),
            ("agent/dev", "device", None),
            ("../traversal.txt", "file", b"evil"),
            ("verifier/reward.json", "file", reward),
            ("verifier/after.txt", "file", b"after the bad members"),
        ])
        with tempfile.TemporaryDirectory() as d:
            dest = pathlib.Path(d) / "dest"
            dest.mkdir()
            skipped, note = p2.extract_session_tar(raw, dest)
            self.assertIsNone(note)
            self.assertEqual((dest / "verifier" / "reward.json").read_bytes(), reward)
            self.assertTrue((dest / "verifier" / "after.txt").exists())
            self.assertFalse((pathlib.Path(d) / "traversal.txt").exists())  # not written outside dest
            self.assertFalse(os.path.lexists(dest / "agent/codex-home/tmp/arg0/codex-arg0X/applypatch"))
            self.assertFalse((dest / "agent" / "escape-link").is_symlink())
            self.assertEqual(len(skipped), 4, skipped)

    def test_truncated_archive_keeps_what_came_before(self):
        raw = make_tar([("verifier/reward.json", "file", b'{"reward": 1}'), ("agent/big", "file", b"x" * 200000)])
        with tempfile.TemporaryDirectory() as d:
            skipped, note = p2.extract_session_tar(raw[:len(raw) // 2], d)
            self.assertIsNotNone(note)
            self.assertTrue((pathlib.Path(d) / "verifier" / "reward.json").exists())


class StartRetry(unittest.TestCase):
    def session(self, fails):
        s = p2.Session.__new__(p2.Session)
        s.task, s.sid, s.start_attempts, s.calls, s.throttles = "pr4121", "p2-a", 1, 0, 0
        s.stopped, seq = [], list(fails)
        def cmd(command, timeout=300, retries=12):
            e = seq.pop(0)
            if e:
                raise e
            return "OK", 0, 1.0
        s.cmd, s.stop = cmd, lambda: s.stopped.append(s.sid)
        return s

    def setUp(self):
        self._sleep, p2.time.sleep = p2.time.sleep, lambda x: None

    def tearDown(self):
        p2.time.sleep = self._sleep

    def test_retries_runtime_client_error_on_a_new_session(self):
        class RuntimeClientError(Exception): pass
        s = self.session([RuntimeClientError("500"), RuntimeClientError("500"), None])
        self.assertEqual(s.start("x")[0], "OK")
        self.assertEqual((s.start_attempts, len(s.stopped), s.sid == "p2-a"), (3, 2, False))

    def test_gives_up_after_three_and_does_not_retry_other_errors(self):
        class RuntimeClientError(Exception): pass
        with self.assertRaises(RuntimeClientError):
            self.session([RuntimeClientError("500")] * 3).start("x")
        s = self.session([ValueError("bad"), None])
        with self.assertRaises(ValueError):
            s.start("x")
        self.assertEqual(s.start_attempts, 1)


class Summary(unittest.TestCase):
    def test_lost_trial_shows_error_and_rerun_hint(self):
        rs = [{"task": "pr371", "trial_name": "t__aaaaaaa", "timings": {}, "reward": {"reward": 1.0}, "errors": []},
              {"task": "pr371", "trial_name": "t__bbbbbbb", "timings": {}, "reward": None,
               "errors": ["INVALID TRIAL (judge): taste judge error: TypeError(\"string indices must be integers, not 'str'\")"]}]
        with tempfile.TemporaryDirectory() as d:
            p2.summarize(rs, d, 1.0)
            txt = (pathlib.Path(d) / "summary.md").read_text()
        self.assertIn("TypeError", txt)
        self.assertIn("1 of 2 trials have no reward", txt)
        self.assertIn("run pr371:1", txt)


if __name__ == "__main__":
    unittest.main()
