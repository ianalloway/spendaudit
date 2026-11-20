"""Offline call accounting for the judge runner."""
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

from src import run_judges as judges


class JudgeBudgetTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(judges.MODELS, {"demo": {"provider": "openai", "reasoning": False, "rpm": 60000}}, clear=True))
        stack.enter_context(patch.dict(judges._last, {}, clear=True))
        stack.enter_context(patch.dict(judges.os.environ, {"OPENAI_API_KEY": "unused"}))
        stack.enter_context(patch.object(judges, "calls", 0))
        self.send = stack.enter_context(patch.object(judges.urllib.request, "urlopen"))

    def test_first_attempt_succeeds(self):
        self.send.return_value = io.BytesIO(b'{"ok": true}')
        self.assertEqual(judges.call("demo", []), {"ok": True})
        self.assertEqual(judges.calls, 1)

    def test_retry_consumes_a_second_slot(self):
        self.send.side_effect = [TimeoutError(), io.BytesIO(b'{"ok": true}')]
        with patch.object(judges.time, "sleep"):
            self.assertEqual(judges.call("demo", []), {"ok": True})
        self.assertEqual(judges.calls, 2)

    def test_cap_blocks_retry_and_other_workers(self):
        self.send.side_effect = TimeoutError()
        with patch.object(judges, "MAX_CALLS", 1), patch.object(judges.time, "sleep"):
            with ThreadPoolExecutor(4) as pool:
                futures = [pool.submit(judges.call, "demo", []) for _ in range(4)]
                for future in futures:
                    with self.assertRaisesRegex(RuntimeError, "call cap"):
                        future.result()
        self.assertEqual(self.send.call_count, 1)

    def test_zero_limit_runs_no_judgments(self):
        with patch.object(judges.sys, "argv", ["run_judges", "--limit", "0"]):
            with patch.object(judges, "run_one") as job, redirect_stdout(io.StringIO()):
                judges.main()
        job.assert_not_called()

    def test_negative_limit_fails_before_processing(self):
        with patch.object(judges.sys, "argv", ["run_judges", "--limit", "-1"]):
            with self.assertRaisesRegex(SystemExit, "--limit"):
                judges.main()
