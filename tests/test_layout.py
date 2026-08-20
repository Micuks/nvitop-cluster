import unittest

from nvitop_cluster.nvitop_cluster import _history_key, _update_history, render_dashboard


def fake_results(hosts=8, gpus_per_host=8):
    results = []
    for host_index in range(hosts):
        gpus = []
        procs = []
        for gpu_index in range(gpus_per_host):
            gpus.append(
                {
                    "index": gpu_index,
                    "name": "NVIDIA H800",
                    "util": 30 + gpu_index * 9,
                    "mem_used": 35840 + gpu_index * 512,
                    "mem_total": 81920,
                    "temp": 39 + gpu_index,
                    "power": 140 + gpu_index * 20,
                }
            )
            procs.append(
                {
                    "gpu_index": gpu_index,
                    "pid": 100000 + host_index * 10 + gpu_index,
                    "user": "root",
                    "cmdline": (
                        "/long/env/bin/python app/video_temporal/launch.py "
                        "--fname /very/long/run/name/that/must/not/wrap"
                    ),
                    "resolved": True,
                }
            )
        suffix = " (local)" if host_index == 0 else ""
        results.append(
            (f"10.48.40.{90 + host_index}{suffix}", {"gpus": gpus, "procs": procs}, None)
        )
    return results


class LayoutTests(unittest.TestCase):
    def test_auto_layout_fits_eight_by_eight_in_215_by_58(self):
        text = render_dashboard(
            fake_results(),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=215,
            rows=58,
            verbose=False,
            cmd_align="left",
            layout="auto",
            selected_host=0,
            attention_only=False,
        )
        lines = text.splitlines()
        self.assertLessEqual(len(lines), 58)
        self.assertLessEqual(max(map(len, lines)), 215)
        self.assertIn("10.48.40.90", text)
        self.assertIn("10.48.40.97", text)
        self.assertIn("CMD ×8", text)
        self.assertIn("UTIL", text)
        self.assertIn("VRAM", text)
        self.assertNotIn("#100000", text)
        self.assertIn("OVERVIEW/COMPACT", text)
        self.assertNotIn("HISTORY", text)
        self.assertEqual(lines[1].count("╮"), 2)

    def test_detail_layout_shows_only_selected_host(self):
        text = render_dashboard(
            fake_results(),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=215,
            rows=58,
            verbose=False,
            cmd_align="left",
            layout="detail",
            selected_host=3,
            attention_only=False,
        )
        self.assertIn("10.48.40.93", text)
        self.assertIn("100030", text)
        self.assertNotIn("10.48.40.92", text)
        self.assertNotIn("10.48.40.94", text)
        self.assertLessEqual(max(map(len, text.splitlines())), 215)

    def test_wide_terminal_uses_four_by_two_and_full_width(self):
        text = render_dashboard(
            fake_results(),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=319,
            rows=77,
            verbose=False,
            cmd_align="left",
            layout="auto",
            selected_host=0,
            attention_only=False,
        )
        lines = text.splitlines()
        self.assertEqual(lines[1].count("╭─"), 4)
        self.assertEqual(max(map(len, lines)), 319)
        self.assertIn("10.48.40.97", text)
        self.assertIn("OVERVIEW/FULL", text)
        self.assertEqual(text.count("UTIL HISTORY"), 8)
        self.assertEqual(text.count("VRAM HISTORY"), 8)
        self.assertNotIn("trend", text)
        self.assertIn("100┤", text)
        self.assertEqual(lines[1].count("╮"), 4)

    def test_partial_final_row_redistributes_full_width(self):
        text = render_dashboard(
            fake_results(hosts=10),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=319,
            rows=77,
            verbose=False,
            cmd_align="left",
            layout="overview",
            selected_host=0,
            attention_only=False,
        )
        lines = text.splitlines()
        last_host_row = next(line for line in lines if "10.48.40.98" in line)
        self.assertIn("10.48.40.99", last_host_row)
        self.assertEqual(last_host_row.count("╭─"), 2)
        self.assertEqual(len(last_host_row), 319)
        self.assertEqual(max(map(len, lines)), 319)

    def test_sixteen_hosts_fit_as_four_by_four(self):
        text = render_dashboard(
            fake_results(hosts=16),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=319,
            rows=70,
            verbose=False,
            cmd_align="left",
            layout="overview",
            selected_host=0,
            attention_only=False,
        )
        lines = text.splitlines()
        self.assertEqual(lines[1].count("╭─"), 4)
        self.assertIn("10.48.40.105", text)
        self.assertNotIn("page 1/", text)
        self.assertEqual(text.count("UTIL RANGE"), 16)
        self.assertEqual(text.count("THERMAL"), 16)
        self.assertLessEqual(len(lines), 70)
        self.assertEqual(max(map(len, lines)), 319)

    def test_very_wide_terminal_uses_full_host_history(self):
        results = fake_results()
        history = {}
        for _ in range(12):
            _update_history(history, results)
        text = render_dashboard(
            results,
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=400,
            rows=77,
            verbose=False,
            cmd_align="left",
            layout="overview",
            selected_host=0,
            attention_only=False,
            history=history,
        )
        self.assertIn("OVERVIEW/FULL", text)
        self.assertEqual(text.count("UTIL HISTORY"), 8)
        self.assertEqual(text.count("VRAM HISTORY"), 8)
        self.assertIn("████", text)
        self.assertLessEqual(len(text.splitlines()), 77)
        self.assertEqual(max(map(len, text.splitlines())), 400)

    def test_history_is_bounded_and_keyed_by_host_and_gpu(self):
        results = fake_results(hosts=1, gpus_per_host=1)
        history = {}
        for _ in range(5):
            _update_history(history, results, maxlen=3)
        key = _history_key("10.48.40.90 (local)", 0, "util")
        self.assertEqual(len(history[key]), 3)
        host_key = _history_key("10.48.40.90 (local)", -1, "util")
        self.assertEqual(len(history[host_key]), 3)

    def test_attention_layout_reports_all_healthy(self):
        text = render_dashboard(
            fake_results(),
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=215,
            rows=58,
            verbose=False,
            cmd_align="left",
            layout="overview",
            selected_host=0,
            attention_only=True,
        )
        self.assertIn("No GPUs currently require attention", text)

    def test_attention_layout_marks_stalled_gpu(self):
        results = fake_results(hosts=1)
        results[0][1]["gpus"][0]["util"] = 0
        text = render_dashboard(
            results,
            color_on=False,
            show_procs=True,
            cmd_width=0,
            cols=120,
            rows=30,
            verbose=False,
            cmd_align="left",
            layout="overview",
            selected_host=0,
            attention_only=True,
        )
        self.assertIn("⚠ idle", text)
        self.assertNotIn("  1 H800", text)


if __name__ == "__main__":
    unittest.main()
