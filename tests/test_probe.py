import unittest

from nvitop_cluster.nvitop_cluster_probe import _elapsed_from_proc_stat


class ProbeTests(unittest.TestCase):
    def test_elapsed_parser_handles_spaces_and_closing_parenthesis_in_comm(self):
        # Fields after comm start at Linux proc stat field 3; field 22 is starttime.
        fields_3_to_21 = ["S"] + ["0"] * 18
        stat = "123 (worker name) rank) " + " ".join(fields_3_to_21 + ["250", "0"])
        self.assertEqual(_elapsed_from_proc_stat(stat, uptime=10.0, clock_ticks=100), 7.5)


if __name__ == "__main__":
    unittest.main()
