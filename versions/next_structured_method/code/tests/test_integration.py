from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

from fixtures import examples, tiny_config, tiny_model
from trace_structured.artifacts import TeacherCache, atomic_json, atomic_torch, build_cache, file_hash, identity, load_checkpoint
from trace_structured.cli import main, safe_output, verify_lineage
from trace_structured.data import Example
from trace_structured.runner import fit, strict_validate


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="trace-graph-free-tests-")
        cls.root = Path(cls.temp.name)
        cls.config = tiny_config(stage1_epochs=2, stage2_epochs=2, sft_lr=.001, rl_lr=.0001,
                                 sft_noise_start=0., sft_noise_fraction=1.)
        cls.train, cls.val = examples(), examples('val')
        model = tiny_model(cls.config, lora=True)
        cls.identity = identity(cls.config, {"fixture": "random-tiny-qwen-not-a-formal-run"}, model.tokenizer)
        with redirect_stdout(io.StringIO()):
            fit(model, cls.train, cls.val, cls.root / 'stage0', 'stage0', cls.identity)
            cls.stage0 = cls.root / 'stage0/last.ckpt'
            build_cache(tiny_model(cls.config, lora=True), cls.train, cls.root / 'teacher_cache',
                        run_identity=cls.identity, teacher_checkpoint=cls.stage0)
            cls.cache = TeacherCache(cls.root / 'teacher_cache', cls.train, tiny_model(cls.config, lora=True),
                                     cls.identity, file_hash(cls.stage0))
            for stage in ('stage1', 'stage2'):
                parent = cls.stage0 if stage == 'stage1' else cls.root / 'stage1/best.ckpt'
                fit(tiny_model(cls.config, lora=True), cls.train, cls.val, cls.root / stage, stage,
                    cls.identity, cache=cls.cache, parent_checkpoint=parent, gradient_audit_every=1)
            strict_validate(tiny_model(cls.config, lora=True), cls.val, cls.root / 'stage2/best.ckpt',
                            cls.root / 'strict_validation', cls.identity, expected_count=2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_three_stage_lineage_and_five_unique_passes(self):
        manifest = verify_lineage(self.root, self.config, self.val)
        self.assertTrue(manifest['verified'])
        result = json.loads((self.root / 'strict_validation/strict_validation_summary.json').read_text())
        self.assertEqual(len(result['results']), 5)
        self.assertTrue(all(x['count'] == 2 for x in result['results']))
        self.assertEqual(len(set(x['correct'] for x in result['results'])), 1)
        before = file_hash(self.root / 'stage2/best.ckpt')
        strict_validate(tiny_model(self.config, lora=True), self.val, self.root / 'stage2/best.ckpt',
                        self.root / 'strict_validation', self.identity, expected_count=2)
        self.assertEqual(before, file_hash(self.root / 'stage2/best.ckpt'))

    def test_cache_fails_closed_on_wrong_cot_projection_and_identity(self):
        args = (self.root / 'teacher_cache', self.train, tiny_model(self.config, lora=True), self.identity, file_hash(self.stage0))
        changed = [replace(self.train[0], steps=('different CoT', 'same length')), self.train[1]]
        with self.assertRaises(ValueError):
            TeacherCache(args[0], changed, *args[2:])
        args[2].semantic_projection.add_(1)
        with self.assertRaises(ValueError):
            TeacherCache(*args)
        with self.assertRaises(ValueError):
            TeacherCache(args[0], self.train, tiny_model(self.config, lora=True), self.identity, 'wrong-teacher')
        with self.assertRaises(ValueError):
            build_cache(tiny_model(self.config, lora=True), self.val, self.root / 'bad-val-cache',
                        run_identity=self.identity, teacher_checkpoint=self.stage0)
        self.assertFalse((self.root / 'bad-val-cache').exists())
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            (copied / 'manifest.json').write_bytes((args[0] / 'manifest.json').read_bytes())
            (copied / 'targets.pt').write_bytes((args[0] / 'targets.pt').read_bytes() + b'corrupt')
            with self.assertRaises(ValueError):
                TeacherCache(copied, self.train, tiny_model(self.config, lora=True), self.identity, file_hash(self.stage0))

    def test_exact_stochastic_sft_and_rl_resume(self):
        for stage in ('stage1', 'stage2'):
            with self.subTest(stage=stage):
                destination = self.root / f'resume-{stage}'
                parent = self.stage0 if stage == 'stage1' else self.root / 'stage1/best.ckpt'
                with redirect_stdout(io.StringIO()):
                    progress = fit(tiny_model(self.config, lora=True), self.train, self.val, destination, stage,
                                   self.identity, cache=self.cache, parent_checkpoint=parent, max_updates=1,
                                   gradient_audit_every=1)
                    self.assertFalse(progress['complete'])
                    with self.assertRaises(ValueError):
                        load_checkpoint(destination / 'last.ckpt', require_complete=True)
                    random.seed(991)
                    torch.manual_seed(991)
                    # Base is immutable and must be the same; seed divergence applies only to RNG after construction.
                    resumed_model = tiny_model(self.config, lora=True)
                    random.seed(991)
                    torch.manual_seed(991)
                    fit(resumed_model, self.train, self.val, destination, stage, self.identity,
                        cache=self.cache, parent_checkpoint=parent, resume=destination / 'last.ckpt',
                        gradient_audit_every=1)
                full = load_checkpoint(self.root / stage / 'last.ckpt')
                resumed = load_checkpoint(destination / 'last.ckpt')
                for key in full['model']:
                    self.assertTrue(torch.equal(full['model'][key], resumed['model'][key]), f'{stage}: {key}')
                self.assertEqual(full['progress'], resumed['progress'])
                self.assertEqual(full['scheduler'], resumed['scheduler'])
                self.assertTrue(torch.equal(full['rng_by_rank'][0]['torch'], resumed['rng_by_rank'][0]['torch']))

    def test_completed_checkpoint_repairs_missing_finalization_idempotently(self):
        import shutil
        path = self.root / 'finalization-recovery'
        shutil.copytree(self.root / 'stage1', path)
        best = load_checkpoint(path / 'best.ckpt')
        best.pop('stage_run_complete')
        best.pop('stage_completed_at_step')
        atomic_torch(path / 'best.ckpt', best)
        (path / 'training_summary.json').unlink()
        with redirect_stdout(io.StringIO()):
            for _ in range(2):
                fit(tiny_model(self.config, lora=True), self.train, self.val, path, 'stage1', self.identity,
                    cache=self.cache, parent_checkpoint=self.stage0, resume=path / 'last.ckpt')
                current = file_hash(path / 'best.ckpt')
                if _:
                    self.assertEqual(previous, current)
                previous = current
        self.assertTrue(load_checkpoint(path / 'best.ckpt', require_complete=True)['stage_run_complete'])
        self.assertTrue(json.loads((path / 'training_summary.json').read_text())['complete'])

    def test_preflight_and_foreign_checkpoint_protection(self):
        path = self.root / 'invalid-parent'
        with self.assertRaises(ValueError):
            fit(tiny_model(self.config), self.train, self.val, path, 'stage1', self.identity)
        self.assertFalse(path.exists())
        legacy = self.root / 'legacy.ckpt'
        atomic_torch(legacy, {'state_dict': {'weight': torch.ones(1)}})
        with self.assertRaises(ValueError):
            load_checkpoint(legacy)
        with self.assertRaises(ValueError):
            repo = Path(__file__).resolve().parents[4]
            safe_output(repo / 'main/native_v9/new', repo / 'data/GSM8k-Aug-NL')
        with self.assertRaises(ValueError):
            strict_validate(tiny_model(self.config, lora=True), self.train, self.root / 'stage2/best.ckpt',
                            self.root / 'not-validation', self.identity, expected_count=2)

    def test_tampered_epoch_and_pass_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'run'
            shutil.copytree(self.root, copied)
            artifact = copied / 'stage2/validation_epoch1.json'
            original = artifact.read_bytes()
            value = json.loads(original)
            value['records'][1] = value['records'][0]
            atomic_json(artifact, value)
            with self.assertRaises(ValueError):
                verify_lineage(copied, self.config, self.val)
            artifact.write_bytes(original)
            p = copied / 'strict_validation/pass_1.json'
            value = json.loads(p.read_text())
            value['evaluation_identity']['checkpoint_sha256'] = 'wrong'
            atomic_json(p, value)
            with self.assertRaises(ValueError):
                strict_validate(tiny_model(self.config, lora=True), self.val, copied / 'stage2/best.ckpt',
                                copied / 'strict_validation', self.identity, expected_count=2)


class DistributedTests(unittest.TestCase):
    def test_two_rank_cpu_reduction_and_partial_batch_training(self):
        with tempfile.TemporaryDirectory(prefix='trace-ddp-test-') as directory:
            worker = Path(__file__).with_name('distributed_worker.py')
            result = subprocess.run([sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                                     '--nproc_per_node=2', str(worker), directory], capture_output=True,
                                    text=True, timeout=90, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                    'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = tiny_config(global_batch_size=2)
            train = [*examples(), Example('train:2', 'One plus two?', ('1+2=3',), '3')]
            with redirect_stdout(io.StringIO()):
                fit(tiny_model(config, lora=True), train, examples('val'), Path(directory) / 'serial',
                    'stage0', {'fixture': 'distributed'})
            parallel = load_checkpoint(Path(directory) / 'parallel/last.ckpt')
            serial = load_checkpoint(Path(directory) / 'serial/last.ckpt')
            self.assertEqual(parallel['world_size'], 2)
            self.assertEqual(parallel['progress'], serial['progress'])
            for key in serial['model']:
                torch.testing.assert_close(parallel['model'][key], serial['model'][key], atol=2e-7, rtol=2e-6)
            with self.assertRaises(ValueError):
                fit(tiny_model(config, lora=True), train, examples('val'), Path(directory) / 'parallel',
                    'stage0', {'fixture': 'distributed'}, resume=Path(directory) / 'parallel/last.ckpt')


class CliTests(unittest.TestCase):
    def test_pipeline_dry_run_does_not_load_model_start_job_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            text = io.StringIO()
            with redirect_stdout(text), patch('trace_structured.cli.load_model') as model, patch('trace_structured.cli.subprocess.run') as run:
                main(['pipeline', '--output', str(output), '--workers', '2', '--dry-run'])
            model.assert_not_called()
            run.assert_not_called()
            self.assertFalse(output.exists())
            result = json.loads(text.getvalue())
            self.assertEqual(result['data']['val']['count'], 747)
            self.assertEqual(len(result['commands']), 5)
            self.assertNotIn('readcot_qsa_qwen_dc', text.getvalue())


if __name__ == '__main__':
    unittest.main()
