"""Stdlib-only fixtures: no model loading, credentials, network, or live files."""
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ace_step_server as server
from render_receipts import OperationConflict, file_identity


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.runtime = SimpleNamespace(output_dir=self.root, _model_id="synthetic", engine="v15",
                                       dtype="float32", checkpoint_dir="synthetic")
        self.states = []

    def tearDown(self):
        for state in self.states:
            state.receipts.close()
        self.directory.cleanup()

    def state(self):
        with patch.object(threading.Thread, "start"), patch.dict(os.environ, {"ACE_WATCHDOG": "0"}):
            state = server.AppState(self.runtime)
        self.states.append(state)
        return state

    def payload(self):
        return {"operation_id": str(uuid.uuid4()), "prompt": "fixture", "seed": 0}

    def complete(self, state, job):
        path = self.root / f"job_{job.id:06d}.wav"
        path.write_bytes(b"synthetic wav")
        job.audios = [server.AudioRecord(path, 1.0, 48000)]
        job.output_identity = {"wav": file_identity(path)}
        job.status = "done"
        state.receipts.save(asdict(job))
        return path

    def test_repeated_enqueue_conflict_and_restart(self):
        first = self.state()
        payload = self.payload()
        job = first.enqueue(payload)
        self.assertEqual(first.enqueue(payload).id, job.id)
        self.assertEqual(first._queue.qsize(), 1)
        with self.assertRaises(OperationConflict):
            first.enqueue({**payload, "prompt": "different"})
        original_id = job.id
        first.receipts.close()
        restored = self.state()
        self.assertEqual(restored.enqueue(payload).id, original_id)
        self.assertEqual(restored.get(original_id).payload_dict()["seed"], 0)
        self.assertEqual(restored._queue.qsize(), 1)

    def test_running_restart_never_rerenders_accepted_work(self):
        first = self.state()
        payload = self.payload()
        job = first.enqueue(payload)
        job.status = "running"
        first.receipts.save(asdict(job))
        first.receipts.close()
        restored = self.state()
        record = restored.enqueue(payload)
        self.assertEqual(record.id, job.id)
        self.assertEqual(record.status, "interrupted")
        self.assertIn("without resubmission", record.error)
        self.assertTrue(restored._queue.empty())

    def test_completed_restart_and_expiry_preserve_receipt_and_manifest(self):
        first = self.state()
        payload = self.payload()
        job = first.enqueue(payload)
        path = self.complete(first, job)
        digest = job.request_sha256
        first.receipts.close()
        restored = self.state()
        same = restored.enqueue(payload)
        self.assertEqual(same.payload_dict()["status"], "done")
        self.assertEqual(same.output_identity, job.output_identity)
        self.assertIn(job.id, restored.receipts.retained_ids())
        with self.assertRaises(OperationConflict):
            restored.receipts.acknowledge(payload["operation_id"], "wrong")
        restored.receipts.acknowledge(payload["operation_id"], digest)
        self.assertNotIn(job.id, restored.receipts.retained_ids())
        path.unlink()
        self.assertEqual(restored.enqueue(payload).payload_dict()["status"], "output_expired")
        self.assertTrue(restored._queue.empty())

    def test_legacy_requests_keep_original_age_retention(self):
        state = self.state()
        job = state.enqueue({"prompt": "legacy"})
        self.complete(state, job)
        self.assertNotIn(job.id, state.receipts.retained_ids())

    def test_pruner_preserves_unacknowledged_outputs(self):
        state = self.state()
        job = state.enqueue(self.payload())
        path = self.complete(state, job)
        old = time.time() - 3600
        os.utime(path, (old, old))
        runtime = server.AceRuntime.__new__(server.AceRuntime)
        runtime.output_dir, runtime.receipts = self.root, state.receipts
        callbacks = []
        with patch.object(threading, "Thread", side_effect=lambda **kwargs: SimpleNamespace(start=lambda: callbacks.append(kwargs["target"]))):
            runtime._start_output_pruner()
        with patch.object(server.time, "sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                callbacks[0]()
        self.assertTrue(path.exists())
        state.receipts.acknowledge(job.operation_id, job.request_sha256)
        with patch.object(server.time, "sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                callbacks[0]()
        self.assertFalse(path.exists())

    def test_http_duplicate_conflict_and_ack_contract(self):
        state = self.state()
        payload = self.payload()

        def post(path, body):
            handler = server.Handler.__new__(server.Handler)
            handler.path = path
            raw = json.dumps(body).encode()
            handler.headers, handler.rfile = {"Content-Length": str(len(raw))}, BytesIO(raw)
            result = []
            handler._json = lambda data, status=200: result.append((status, data))
            with patch.object(server, "STATE", state):
                handler.do_POST()
            return result[0]

        status, receipt = post("/generate/enqueue", payload)
        self.assertEqual(status, 200)
        self.assertEqual(receipt["operation_id"], payload["operation_id"])
        self.assertEqual(post("/generate/enqueue", payload)[1]["id"], receipt["id"])
        self.assertEqual(post("/generate/enqueue", {**payload, "seed": 1})[0], 409)
        self.complete(state, state.get(receipt["id"]))
        self.assertEqual(post(f"/operations/{payload['operation_id']}/ack", {"request_sha256": receipt["request_sha256"]})[0], 200)

    def test_generate_passes_generator_only_for_explicit_seeds(self):
        runtime = server.AceRuntime.__new__(server.AceRuntime)
        runtime.output_dir, runtime.engine, runtime._lock = self.root, "v15", threading.Lock()
        captured = []

        class Audio:
            @property
            def T(self): return self
            def cpu(self): return self
            def float(self): return self
            def numpy(self): return []

        class Pipeline:
            sample_rate = 48000
            def __call__(self, **kwargs):
                captured.append(kwargs)
                return SimpleNamespace(audios=[Audio()])

        runtime._load = lambda: Pipeline()
        fake_torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: ("seed", seed)))
        fake_soundfile = SimpleNamespace(write=lambda path, *args: Path(path).write_bytes(b"wav"))
        with patch.dict("sys.modules", {"torch": fake_torch, "soundfile": fake_soundfile}), \
             patch.object(server, "_to_pcm16"), patch.object(server, "_wav_info", return_value=(1, 48000)), \
             patch.object(server, "_waveform_png", side_effect=lambda wav, target: target.write_bytes(b"png")):
            runtime.generate(server.Job(1, {"prompt": "legacy"}))
            runtime.generate(server.Job(2, {"prompt": "seeded", "seed": 0}))
        self.assertNotIn("generator", captured[0])
        self.assertEqual(captured[1]["generator"], ("seed", 0))
        self.assertEqual(captured[1]["num_inference_steps"], captured[0]["num_inference_steps"])

    def test_seed_only_changes_explicitly_seeded_calls(self):
        self.assertEqual(server._pipeline_seed({}), {})
        seeds = []
        fake_torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seeds.append((kwargs, seed)) or "generator"))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(server._pipeline_seed({"seed": 0}), {"generator": "generator"})
        self.assertEqual(seeds, [({"device": "cuda"}, 0)])


if __name__ == "__main__":
    unittest.main()
