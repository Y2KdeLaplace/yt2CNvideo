from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from videodub.ui import VideoDubApp


class VideoSelectionTests(unittest.TestCase):
    def test_control_click_adds_an_unselected_row(self) -> None:
        tree = Mock()
        tree.identify_row.return_value = "2"
        tree.selection.return_value = ("1",)
        app = SimpleNamespace(job_tree=tree)

        result = VideoDubApp._toggle_job_selection(
            app,
            SimpleNamespace(y=20),
        )

        self.assertEqual(result, "break")
        tree.selection_add.assert_called_once_with("2")
        tree.focus.assert_called_once_with("2")

    def test_control_click_removes_a_selected_row(self) -> None:
        tree = Mock()
        tree.identify_row.return_value = "2"
        tree.selection.return_value = ("1", "2")
        app = SimpleNamespace(job_tree=tree)

        VideoDubApp._toggle_job_selection(app, SimpleNamespace(y=20))

        tree.selection_remove.assert_called_once_with("2")

    def test_select_all_requires_an_existing_selection(self) -> None:
        tree = Mock()
        tree.selection.return_value = ("1",)
        tree.get_children.return_value = ("0", "1", "2")
        app = SimpleNamespace(job_tree=tree)

        result = VideoDubApp._select_all_jobs(app)

        self.assertEqual(result, "break")
        tree.selection_set.assert_called_once_with(("0", "1", "2"))


if __name__ == "__main__":
    unittest.main()
