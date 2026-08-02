from __future__ import annotations

import unittest
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from videodub.ui import VideoDubApp
from videodub.config import AppConfig
from videodub.media import VideoJob


class VideoSelectionTests(unittest.TestCase):
    def test_parallel_failure_does_not_cancel_other_jobs(self) -> None:
        events = queue.Queue()
        completed: list[str] = []

        def process_one(_config, job, _stages, _slot):
            if job.title == "out-of-memory":
                raise MemoryError("model allocation failed")
            completed.append(job.title)

        app = SimpleNamespace(
            events=events,
            _process_one_job=Mock(side_effect=process_one),
            _cancel_active_runners=Mock(),
        )
        jobs = [
            VideoJob(Path("failed.mp4"), title="out-of-memory"),
            VideoJob(Path("continued.mp4"), title="continued"),
        ]

        VideoDubApp._processing_worker(
            app,
            AppConfig(),
            jobs,
            (False, False, False, False),
            2,
        )

        messages = []
        while not events.empty():
            messages.append(events.get_nowait())
        self.assertEqual(completed, ["continued"])
        self.assertFalse(any(kind == "error" for kind, _value in messages))
        self.assertTrue(
            any(
                kind == "log" and "out-of-memory" in str(value)
                for kind, value in messages
            )
        )
        self.assertIn(("done", "处理完成，1 个任务失败"), messages)
        app._cancel_active_runners.assert_not_called()

    def test_extract_stage_skips_a_subtitle_only_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Lecture.en.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello.\n",
                encoding="utf-8",
            )
            events = queue.Queue()
            app = SimpleNamespace(
                events=events,
                runners_lock=threading.Lock(),
                active_runners=[],
                session_api_key="",
            )
            job = VideoJob(
                root / "Lecture.mp4",
                title="Lecture",
                source_subtitle_path=source,
            )

            VideoDubApp._process_one_job(
                app,
                AppConfig(work_dir=str(root)),
                job,
                (True, False, False, False),
                0,
            )

            messages = []
            while not events.empty():
                messages.append(events.get_nowait())
            self.assertTrue(
                any("已跳过语音提取" in str(message) for message in messages)
            )

    def test_entering_process_tab_clears_the_selection(self) -> None:
        tree = Mock()
        tree.selection.return_value = ("1", "2")
        app = SimpleNamespace(
            job_tree=tree,
            notebook=SimpleNamespace(select=Mock(return_value=".process")),
            process_tab=".process",
            _refresh_jobs=Mock(),
            after_idle=Mock(),
            _resize_notebook_to_current_tab=Mock(),
            focus_set=Mock(),
        )

        VideoDubApp._on_tab_changed(app)

        app._refresh_jobs.assert_called_once_with()
        tree.selection_remove.assert_called_once_with("1", "2")

    def test_clicking_blank_space_clears_the_selection(self) -> None:
        tree = Mock()
        tree.identify_row.return_value = ""
        tree.selection.return_value = ("1", "2")
        app = SimpleNamespace(job_tree=tree)

        result = VideoDubApp._clear_job_selection_on_blank(
            app,
            SimpleNamespace(y=200),
        )

        self.assertEqual(result, "break")
        tree.selection_remove.assert_called_once_with("1", "2")

    def test_clicking_a_row_keeps_the_default_single_select_behavior(self) -> None:
        tree = Mock()
        tree.identify_row.return_value = "2"
        app = SimpleNamespace(job_tree=tree)

        result = VideoDubApp._clear_job_selection_on_blank(
            app,
            SimpleNamespace(y=20),
        )

        self.assertIsNone(result)
        tree.selection_remove.assert_not_called()

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

    def test_select_all_does_nothing_without_an_existing_selection(self) -> None:
        tree = Mock()
        tree.selection.return_value = ()
        app = SimpleNamespace(job_tree=tree)

        result = VideoDubApp._select_all_jobs(app)

        self.assertEqual(result, "break")
        tree.selection_set.assert_not_called()

    def test_selected_jobs_follow_table_row_order(self) -> None:
        jobs = [SimpleNamespace(title=title) for title in ("first", "second", "third")]
        tree = Mock()
        tree.selection.return_value = ("2", "0")
        tree.get_children.return_value = ("0", "1", "2")
        app = SimpleNamespace(job_tree=tree, jobs=jobs)

        selected = VideoDubApp._selected_jobs(app)

        self.assertEqual([job.title for job in selected], ["first", "third"])

    def test_processing_without_parallel_runs_jobs_in_order(self) -> None:
        processed: list[str] = []
        jobs = [VideoJob(Path(f"{title}.mp4"), title=title) for title in ("a", "b", "c")]
        app = SimpleNamespace(
            events=queue.Queue(),
            _process_one_job=Mock(
                side_effect=lambda _config, job, _stages, _slot: processed.append(
                    job.title
                )
            ),
            _cancel_active_runners=Mock(),
        )

        VideoDubApp._processing_worker(
            app,
            AppConfig(),
            jobs,
            (False, False, False, False),
            1,
        )

        self.assertEqual(processed, ["a", "b", "c"])

    def test_download_state_locks_only_the_url_input(self) -> None:
        app = SimpleNamespace(
            current_task="",
            url_text=Mock(),
            refresh_jobs_button=Mock(),
            job_tree=Mock(),
            download_button=Mock(),
            process_button=Mock(),
            _stop=Mock(),
        )

        VideoDubApp._set_running(app, True, "download")

        app.url_text.configure.assert_called_once_with(state="disabled")
        app.refresh_jobs_button.configure.assert_called_once_with(state="normal")
        app.job_tree.state.assert_called_once_with(("!disabled",))

    def test_process_state_locks_video_selection_and_restores_it_when_done(self) -> None:
        app = SimpleNamespace(
            current_task="",
            url_text=Mock(),
            refresh_jobs_button=Mock(),
            job_tree=Mock(),
            download_button=Mock(),
            process_button=Mock(),
            _stop=Mock(),
            _start_download=Mock(),
            _start_processing=Mock(),
        )

        VideoDubApp._set_running(app, True, "process")
        VideoDubApp._set_running(app, False)

        self.assertEqual(
            app.refresh_jobs_button.configure.call_args_list,
            [unittest.mock.call(state="disabled"), unittest.mock.call(state="normal")],
        )
        self.assertEqual(
            app.job_tree.state.call_args_list,
            [unittest.mock.call(("disabled",)), unittest.mock.call(("!disabled",))],
        )


if __name__ == "__main__":
    unittest.main()
