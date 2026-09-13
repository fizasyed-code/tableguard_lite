from pathlib import Path
import tempfile
import unittest
from tableguard.phase2.scene import scan, load, new_run_dir, sha256


class SceneFileTests(unittest.TestCase):
    def test_scanner_distinguishes_entry_include_urdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'scene.xml').write_text('<mujoco model="unit-test"/>')
            (p/'piece.xml').write_text('<mujocoinclude/>')
            (p/'urdf.xml').write_text('<robot/>')
            rows = scan(p)['xml_files']
            self.assertEqual(sum(r.get('entry_candidate', False) for r in rows), 1)
            self.assertEqual({r['root_tag'] for r in rows}, {'mujoco','mujocoinclude','robot'})

    def test_malformed_xml_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/'bad.xml').write_text('<mujoco>')
            self.assertEqual(scan(p)['xml_files'][0]['root_tag'], 'parse_error')

    def test_xml_entities_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/'bad.xml').write_text('<!DOCTYPE a><mujoco/>')
            self.assertFalse(scan(p)['xml_files'][0]['entry_candidate'])

    def test_missing_directory_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            scan(Path('/definitely_absent_tableguard_scene_folder'))

    def test_empty_folder_not_fake_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(scan(Path(tmp))['xml_files'], [])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_scan_is_bounded_and_labels_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(3): (p/f'{i}.xml').write_text('<mujoco/>')
            report = scan(p, 1)
            self.assertTrue(report['truncated'])
            self.assertEqual(len(report['xml_files']), 1)

    def test_excludes_virtual_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/'.venv').mkdir(); (p/'.venv/test.xml').write_text('<mujoco/>')
            self.assertEqual(scan(p)['xml_files'], [])

    def test_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = new_run_dir(Path(tmp), 'test'); b = new_run_dir(Path(tmp), 'test')
            self.assertNotEqual(a, b)
            self.assertTrue(a.is_dir() and b.is_dir())

    def test_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'file'; p.write_text('a'); first = sha256(p)
            p.write_text('b'); self.assertNotEqual(first, sha256(p))

    def test_no_scene_fallback_before_mujoco_import(self):
        with self.assertRaises(FileNotFoundError):
            load(Path('/no_file_123456.xml'))
