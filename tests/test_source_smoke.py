"""Offline checks: no API calls, gallery data or model downloads."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SourceSmokeTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob('*.py'):
            if not any(p in path.parts for p in ('node_modules', '.venv')):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding='utf-8-sig'))

    def test_notebooks_have_no_outputs(self):
        for path in (ROOT / 'experiments').glob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                self.assertFalse(cell.get('outputs'))
                self.assertFalse(cell.get('attachments'))
                self.assertIsNone(cell.get('execution_count'))

    def test_pipeline_modules_resolve_without_embedded_blobs(self):
        runner = load('lumina_pipeline_test', ROOT / 'ai_pipeline/integrated_server.py')
        for group in (runner.EMBEDDED_SOURCES, runner.EMBEDDED_MODULES):
            for name in group:
                text = runner._decode(group, name)
                ast.parse(text)
        paths = runner.get_pipeline_paths('test-photos')
        self.assertTrue(paths['cluster_dir'].endswith(os.path.join('cropped_insight', 'adaface_results')))

    def test_missing_person_age_is_optional(self):
        source = ast.parse((ROOT / 'ai_pipeline/data_agent.py').read_text(encoding='utf-8'))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'build_age_timeline')
        namespace = {'PERSON_ZERO_AGE': None, 'PERSON_ZERO_AGE_YEAR': 2026}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<age-test>', 'exec'), namespace)
        self.assertEqual(namespace['build_age_timeline']([]), {})

    def test_ai_server_import_does_not_start_workers(self):
        before = {t.ident for t in threading.enumerate()}
        with patch('threading.Thread.start', side_effect=AssertionError('Unexpected worker startup')):
            server = load('lumina_ai_test', ROOT / 'ai_server/ai_server.py')
        self.assertEqual(before, {t.ident for t in threading.enumerate()})
        self.assertTrue(server.INTEGRATED_SERVER.is_file())
        self.assertTrue((server.FORGITHUB_DIR / 'inference_one_image.py').is_file())
        response = server.app.test_client().get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['queue_size'], 0)

    def test_search_routes_with_synthetic_results_and_fake_gemini(self):
        # All photo metadata below is invented for this test.
        sys.path.insert(0, str(ROOT / 'backend'))
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'LUMINA_DATA_DIR': temp, 'LUMINA_REQUIRE_AUTH': '0'}):
            app = load('lumina_backend_test', ROOT / 'backend/app.py')
            parsed = {'date_range': {'start': '2020-01-01', 'end': '2020-12-31'},
                      'date_months': [], 'people': [], 'location': None,
                      'object_terms': ['비행기'], 'activity_terms': []}
            rows = [
                {'filename': 'synthetic-a.jpg', 'datetime': '2020-01-01T12:00:00', 'analysis': {'scene': {'description': '비행기 날개'}}},
                {'filename': 'synthetic-b.jpg', 'datetime': '2021-01-01T12:00:00', 'analysis': {'scene': {'description': '비행기 날개'}}},
                {'filename': 'synthetic-c.jpg', 'datetime': '2020-01-01T12:00:00', 'analysis': {'scene': {'description': '나무'}}},
            ]
            def fake_data(path):
                return rows if str(path).endswith('deep_analyses.json') else {}
            def fake_llm(prompt):
                if 'date_range' in prompt:
                    return json.dumps(parsed, ensure_ascii=False)
                return '{"object_match_ids":[0,1]}'
            with patch.object(app, 'safe_load_json', side_effect=fake_data), patch.object(app, '_invoke_llm', side_effect=fake_llm), patch('requests.post', side_effect=AssertionError('Network forbidden')):
                client = app.app.test_client()
                self.assertEqual(client.post('/api/search', json={'query': ''}).status_code, 400)
                result = client.post('/api/search', json={'query': '2020년 비행기 사진'})
                self.assertEqual(result.status_code, 200, result.json)
                self.assertEqual(result.json['parsed']['date_range'], parsed['date_range'])
                self.assertEqual([r['filename'] for r in result.json['results']], ['synthetic-a.jpg'])


if __name__ == '__main__':
    unittest.main()
