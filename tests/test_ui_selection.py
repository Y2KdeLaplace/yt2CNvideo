from __future__ import annotations

import inspect
import unittest
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from videodub.ui import VideoDubApp
from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.runner import CancelledError, ProcessRunner


class VideoSelectionTests(unittest.TestCase):
    @staticmethod
    def _lock_app():
        app = SimpleNamespace(mlx_inference_lock=threading.Lock())
        app._acquire_mlx_inference = lambda runner: (
            VideoDubApp._acquire_mlx_inference(app, runner)
        )
        return app

    def test_parallel_mlx_stages_share_one_inference_slot(self) -> None:
        app = self._lock_app()
        first_inside = threading.Event()
        release_first = threading.Event()
        second_inside = threading.Event()
        active = 0
        peak_active = 0
        state_lock = threading.Lock()

        def stage(is_first: bool) -> None:
            nonlocal active, peak_active
            with VideoDubApp._mlx_inference_slot(
                app,
                ProcessRunner(),
                enabled=True,
            ):
                with state_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                (first_inside if is_first else second_inside).set()
                if is_first:
                    release_first.wait(1)
                with state_lock:
                    active -= 1

        first = threading.Thread(target=stage, args=(True,))
        second = threading.Thread(target=stage, args=(False,))
        first.start()
        self.assertTrue(first_inside.wait(1))
        second.start()
        self.assertFalse(second_inside.wait(0.1))
        release_first.set()
        first.join(1)
        second.join(1)

        self.assertTrue(second_inside.is_set())
        self.assertEqual(peak_active, 1)

    def test_waiting_for_mlx_slot_can_be_cancelled(self) -> None:
        app = self._lock_app()
        app.mlx_inference_lock.acquire()
        waiting = threading.Event()
        runner = ProcessRunner(
            lambda message: waiting.set() if "等待" in message else None
        )
        errors: list[Exception] = []

        def acquire() -> None:
            try:
                VideoDubApp._acquire_mlx_inference(app, runner)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=acquire)
        worker.start()
        self.assertTrue(waiting.wait(1))
        runner.cancel()
        worker.join(1)
        app.mlx_inference_lock.release()

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)

    def test_non_mlx_stage_does_not_wait_for_mlx_slot(self) -> None:
        app = self._lock_app()
        app.mlx_inference_lock.acquire()
        entered = False

        with VideoDubApp._mlx_inference_slot(
            app,
            ProcessRunner(),
            enabled=False,
        ):
            entered = True

        app.mlx_inference_lock.release()
        self.assertTrue(entered)

    def test_gui_heartbeat_normal_does_not_log(self) -> None:
        app = SimpleNamespace(
            _last_gui_heartbeat=10.0,
            _gui_stall_started_at=None,
            _gui_stall_detected=False,
            _emit_watchdog_event=Mock(),
        )

        VideoDubApp._check_gui_heartbeat(app, now=12.9)

        app._emit_watchdog_event.assert_not_called()

    def test_gui_stall_logs_once_and_then_logs_recovery(self) -> None:
        app = SimpleNamespace(
            _last_gui_heartbeat=10.0,
            _gui_stall_started_at=None,
            _gui_stall_detected=False,
            _emit_watchdog_event=Mock(),
        )

        VideoDubApp._check_gui_heartbeat(app, now=13.4)
        VideoDubApp._check_gui_heartbeat(app, now=15.0)
        self.assertEqual(app._emit_watchdog_event.call_count, 1)
        self.assertIn("stalled for 3.4 s", app._emit_watchdog_event.call_args.args[0])

        app._last_gui_heartbeat = 15.1
        VideoDubApp._check_gui_heartbeat(app, now=15.2)

        self.assertEqual(app._emit_watchdog_event.call_count, 2)
        self.assertIn("recovered after 5.2 s", app._emit_watchdog_event.call_args.args[0])

    def test_watchdog_thread_body_does_not_call_tk(self) -> None:
        source = inspect.getsource(VideoDubApp._watch_gui_heartbeat)
        self.assertNotIn("self.after", source)
        self.assertNotIn("self.update", source)
        self.assertNotIn("self.destroy", source)

    def test_on_close_stops_watchdog_before_destroy(self) -> None:
        calls: list[str] = []
        app = SimpleNamespace(
            model_download_runner=None,
            download_worker=None,
            process_worker=None,
            _persist_config=Mock(),
            _stop_gui_watchdog=Mock(side_effect=lambda: calls.append("watchdog")),
            destroy=Mock(side_effect=lambda: calls.append("destroy")),
        )

        VideoDubApp._on_close(app)

        self.assertEqual(calls, ["watchdog", "destroy"])

    def test_stop_watchdog_sets_event_and_joins_thread(self) -> None:
        stop = threading.Event()
        thread = Mock()
        app = SimpleNamespace(
            _watchdog_stop=stop,
            _watchdog_thread=thread,
            _gui_stall_detected=True,
        )

        VideoDubApp._stop_gui_watchdog(app)

        self.assertTrue(stop.is_set())
        thread.join.assert_called_once_with(timeout=2)

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
        self.assertFalse(any(kind == "task_error" for kind, _value in messages))
        self.assertTrue(
            any(
                kind == "log" and "out-of-memory" in str(value)
                for kind, value in messages
            )
        )
        self.assertIn(
            ("task_done", ("process", "处理完成，1 个任务失败")),
            messages,
        )
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
            download_running=False,
            process_running=False,
            url_text=Mock(),
            refresh_jobs_button=Mock(),
            job_tree=Mock(),
            download_button=Mock(),
            process_button=Mock(),
            _stop_download=Mock(),
            _stop_processing=Mock(),
            _start_download=Mock(),
            _start_processing=Mock(),
        )

        VideoDubApp._set_running(app, True, "download")

        app.url_text.configure.assert_called_once_with(state="disabled")
        app.refresh_jobs_button.configure.assert_called_once_with(state="normal")
        app.job_tree.state.assert_called_once_with(("!disabled",))
        self.assertEqual(app.process_button.configure.call_args.kwargs["state"], "normal")

    def test_process_state_locks_video_selection_and_restores_it_when_done(self) -> None:
        app = SimpleNamespace(
            download_running=False,
            process_running=False,
            url_text=Mock(),
            refresh_jobs_button=Mock(),
            job_tree=Mock(),
            download_button=Mock(),
            process_button=Mock(),
            _stop_download=Mock(),
            _stop_processing=Mock(),
            _start_download=Mock(),
            _start_processing=Mock(),
        )

        VideoDubApp._set_running(app, True, "process")
        VideoDubApp._set_running(app, False, "process")

        self.assertEqual(
            app.refresh_jobs_button.configure.call_args_list,
            [unittest.mock.call(state="disabled"), unittest.mock.call(state="normal")],
        )
        self.assertEqual(
            app.job_tree.state.call_args_list,
            [unittest.mock.call(("disabled",)), unittest.mock.call(("!disabled",))],
        )

    def test_download_and_processing_states_are_independent(self) -> None:
        app = SimpleNamespace(
            download_running=False,
            process_running=False,
            url_text=Mock(),
            refresh_jobs_button=Mock(),
            job_tree=Mock(),
            download_button=Mock(),
            process_button=Mock(),
            _stop_download=Mock(),
            _stop_processing=Mock(),
            _start_download=Mock(),
            _start_processing=Mock(),
        )

        VideoDubApp._set_running(app, True, "process")
        VideoDubApp._set_running(app, True, "download")

        self.assertTrue(app.download_running)
        self.assertTrue(app.process_running)
        self.assertIs(
            app.download_button.configure.call_args.kwargs["command"],
            app._stop_download,
        )
        self.assertIs(
            app.process_button.configure.call_args.kwargs["command"],
            app._stop_processing,
        )

        VideoDubApp._set_running(app, False, "download")

        self.assertFalse(app.download_running)
        self.assertTrue(app.process_running)
        self.assertEqual(app.job_tree.state.call_args.args[0], ("disabled",))

    def test_running_processing_does_not_block_starting_a_download(self) -> None:
        process_worker = SimpleNamespace(is_alive=lambda: True)
        thread = Mock()
        app = SimpleNamespace(
            process_worker=process_worker,
            download_worker=None,
            url_text=Mock(),
            _sync_basic_config=Mock(return_value=AppConfig()),
            _separator=Mock(),
            download_runner=Mock(),
            _set_running=Mock(),
            _download_worker=Mock(),
        )
        app.url_text.get.return_value = "https://youtu.be/example\n"

        with patch("videodub.ui.threading.Thread", return_value=thread):
            VideoDubApp._start_download(app)

        thread.start.assert_called_once_with()
        app._set_running.assert_called_once_with(True, "download")


if __name__ == "__main__":
    unittest.main()
