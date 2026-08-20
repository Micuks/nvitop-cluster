import unittest

from nvitop_cluster.nvitop_cluster import render_dashboard


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
        self.assertIn("U ███", text)
        self.assertIn("M ███", text)
        self.assertNotIn("#100000", text)

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
