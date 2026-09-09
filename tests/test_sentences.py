import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from videodub.config import AppConfig
from videodub.media import VideoJob
from videodub.openai_compatible import ChatResult
from videodub.runner import ProcessRunner
from videodub.sentences import (
    SentenceUnit, TimelineError, build_sentence_units, display_cues, read_units,
    write_units, spoken_text, validate_timeline,
)
from videodub.subtitles import Cue, write_srt, read_srt
from videodub.subtitle_workflow import SubtitleRepairWorkflow


class TimelineTests(unittest.TestCase):
    def test_eighty_millisecond_spoken_cue_is_valid(self):
        cue = Cue(1, 1280, 1360, "You")

        validate_timeline([cue])
        units = build_sentence_units([cue])

        self.assertEqual(
            [(unit.start_ms, unit.end_ms, unit.text) for unit in units],
            [(1280, 1360, "You")],
        )

    def test_timestamp_regression_and_overlap_include_text_and_time(self):
        messages = []
        with self.assertRaisesRegex(TimelineError, "non_monotonic_timeline"):
            build_sentence_units([Cue(87, 256720, 256721, "Isn't"),
                                  Cue(88, 244164, 244564, "it?")], messages.append)
        self.assertTrue(any("cue 88" in m and "244164" in m and "it?" in m for m in messages))
        self.assertTrue(any("abnormal_overlap" in m for m in messages))

    def test_zero_duration_spoken_cue_cannot_become_sentence_unit(self):
        cue = Cue(1, 10_320, 10_320, "Yet")
        with self.assertRaisesRegex(TimelineError, "invalid_duration"):
            build_sentence_units([cue])
        self.assertEqual(cue, Cue(1, 10_320, 10_320, "Yet"))

    def test_punctuation_and_sound_are_never_spoken(self):
        for text in [".", "。", "”", "……", '.”', "[fly buzzing]", "[wrapper crinkles]", "[stammers]", "[grunts]"]:
            self.assertEqual(spoken_text(text), "")
            units = build_sentence_units([Cue(1, 0, 1000, text)])
            self.assertFalse(any(u.kind == "spoken" for u in units))
        self.assertEqual(spoken_text("  Ｈｉ  [fly buzzing] \n there! "), "Hi there!")

    def test_large_gap_stops_grouping_without_expanding_window(self):
        units = build_sentence_units([Cue(1, 0, 500, "Hello"), Cue(2, 7500, 8200, "there.")])
        self.assertEqual([(u.start_ms, u.end_ms) for u in units], [(0, 500), (7500, 8200)])

    def test_window_limit_needs_real_boundary(self):
        with self.assertRaisesRegex(TimelineError, "sentence_window_too_long"):
            build_sentence_units([Cue(1, 0, 8000, "A long"), Cue(2, 8000, 16000, "sentence.")])
        with self.assertRaisesRegex(TimelineError, "sentence_window_too_long"):
            build_sentence_units([Cue(1, 0, 40000, "One sentence.")])

    def test_final_srt_rejects_bad_timing_and_punctuation_before_writing(self):
        for cues in [[Cue(1, 1, 1, "Hi")],
                     [Cue(1, 0, 1000, ".")], [Cue(1, 1000, 2000, "Hi"), Cue(2, 0, 800, "No")]]:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "out.srt"
                with self.assertRaises(TimelineError):
                    write_srt(path, cues)
                self.assertFalse(path.exists())

    def test_display_changes_cannot_change_tts_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.srt"
            units = build_sentence_units([Cue(1, 0, 1000, "Hello,"), Cue(2, 1000, 2200, "world!")])
            write_units(path, units)
            # A separate presentation layer may split display text at real boundaries.
            write_srt(path, [Cue(1, 0, 1000, "Hello,"), Cue(2, 1000, 2200, "world!")])
            self.assertEqual(read_units(path), units)
            self.assertEqual(len(read_units(path)), 1)
            self.assertEqual(len(read_srt(path)), 2)


class SentenceWorkflowTests(unittest.TestCase):
    def workflow(self):
        return SubtitleRepairWorkflow(AppConfig(subtitle_model="test"), ProcessRunner())

    def source(self):
        return [Cue(1, 0, 800, "Give him"), Cue(2, 800, 2000, "a salute."),
                Cue(3, 2500, 3500, "Understatement.")]

    def test_repair_translation_preserve_identity_and_no_segmentation(self):
        workflow = self.workflow()
        calls = []
        def respond(system, prompt, **kwargs):
            payload = json.loads(prompt)
            calls.append(payload)
            field = "corrected_text" if "校对专家" in system else "translated_text"
            texts = ["Give him a salute.", "Understatement."] if field == "corrected_text" else ["我向他敬了个礼。", "说得太轻了。"]
            return [{"group_id": u["group_id"], field: texts[u["group_id"]-1]} for u in payload["source_sentence_groups"]]
        workflow._call_array = respond
        source = build_sentence_units(self.source())
        corrected, _ = workflow.repair(self.source(), self.source(), {})
        translated = workflow.translate(corrected, {})
        self.assertEqual(len(calls), 2)
        for original, result in zip(source, translated):
            self.assertEqual(replace(result, text=original.text), original)
        self.assertEqual(translated[0].text, "我向他敬了个礼。")
        self.assertEqual(translated[0].last_cue, 2)
        self.assertTrue(all("parts" not in json.dumps(c) for c in calls))

    def test_missing_groups_retry_only_missing_with_full_context(self):
        workflow = self.workflow()
        units = build_sentence_units(self.source())
        workflow._call_array = Mock(side_effect=[
            [{"group_id": 1, "translated_text": "敬礼。"}],
            [{"group_id": 2, "translated_text": "太轻了。"}],
        ])
        result = workflow.translate(units, {})
        request = json.loads(workflow._call_array.call_args_list[1].args[1])
        self.assertEqual([g["group_id"] for g in request["source_sentence_groups"]], [2])
        self.assertIn("Give him a salute.", request["complete_corrected_transcript"])
        self.assertEqual(len(result), 2)

    def test_duplicate_added_empty_and_punctuation_outputs_fail(self):
        for rows in [[{"group_id": 1, "translated_text": "好"}]*2,
                     [{"group_id": 9, "translated_text": "好"}],
                     [{"group_id": 1, "translated_text": ""}],
                     [{"group_id": 1, "translated_text": "。"}]]:
            workflow = self.workflow()
            workflow._call_array = Mock(return_value=rows)
            with self.assertRaises(RuntimeError):
                workflow.translate(build_sentence_units(self.source()), {})
            self.assertEqual(workflow._call_array.call_count, 3)

    def test_bounded_output_groups_with_full_document_context(self):
        workflow = self.workflow()
        units = build_sentence_units([Cue(i+1, i*1000, i*1000+800, f"Sentence {i}.") for i in range(60)])
        calls = []
        def respond(system, prompt, **kwargs):
            request = json.loads(prompt)
            calls.append(request)
            return [{"group_id": g["group_id"], "translated_text": "一句话。"} for g in request["source_sentence_groups"]]
        workflow._call_array = respond
        self.assertEqual(len(workflow.translate(units, {})), 60)
        self.assertEqual([len(c["source_sentence_groups"]) for c in calls], [24, 24, 12])
        self.assertTrue(all("Sentence 59." in c["complete_corrected_transcript"] for c in calls))

    def test_process_job_persists_canonical_units_and_monotonic_srt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "video.en.srt"
            write_srt(source_path, self.source())
            job = VideoJob(root / "video.mp4", source_subtitle_path=source_path, generated_dir=root / "out")
            workflow = self.workflow()
            def chat(system, prompt, **kwargs):
                if "专业术语分析专家" in system:
                    return ChatResult('{}', 0, 0)
                payload = json.loads(prompt)
                field = "corrected_text" if "校对专家" in system else "translated_text"
                return ChatResult(json.dumps([{ "group_id": g["group_id"], field: g["text"] if field == "corrected_text" else "你好。"} for g in payload["source_sentence_groups"]]), 1, 1)
            workflow.client.chat = Mock(side_effect=chat)
            workflow.process_job(job)
            translated = read_units(job.translated_subtitle_path("Chinese"))
            corrected = read_units(job.corrected_subtitle_path)
            self.assertEqual(len(translated), 2)
            self.assertEqual([(u.start_ms, u.end_ms) for u in translated], [(u.start_ms,u.end_ms) for u in corrected])
            validate_timeline(read_srt(job.translated_subtitle_path("Chinese")))

    def test_bad_source_fails_before_any_model_call(self):
        workflow = self.workflow()
        workflow.client.chat = Mock()
        with self.assertRaises(TimelineError):
            workflow.repair(self.source(), [Cue(1, 0, 0, "You")], {})
        workflow.client.chat.assert_not_called()

    def test_malformed_model_json_is_bounded_and_cancellation_is_immediate(self):
        from videodub.runner import CancelledError
        workflow = self.workflow()
        workflow.client.chat = Mock(return_value=ChatResult('invalid', 0, 0))
        with self.assertRaises(RuntimeError):
            workflow.translate(build_sentence_units(self.source()), {})
        self.assertEqual(workflow.client.chat.call_count, 3)
        workflow.client.chat = Mock(side_effect=CancelledError("stopped"))
        with self.assertRaises(CancelledError):
            workflow.translate(build_sentence_units(self.source()), {})
        self.assertEqual(workflow.client.chat.call_count, 1)

    def test_sound_identity_and_non_chinese_language_guidance(self):
        workflow = self.workflow()
        workflow.config.translation_language = "German"
        units = build_sentence_units([Cue(1, 0, 900, "[fly buzzing]"), Cue(2, 1000, 2000, "Hello!")])
        workflow._call_array = Mock(return_value=[{"group_id":1, "translated_text":"[Fliegensummen]"},
                                                  {"group_id":2, "translated_text":"Hallo!"}])
        result = workflow.translate(units, {})
        self.assertEqual(result[0].kind, "sound")
        self.assertEqual(result[1].kind, "spoken")
        self.assertIn("German", workflow._call_array.call_args.args[0])
        self.assertNotIn("自然简体中文", workflow._call_array.call_args.args[0])
