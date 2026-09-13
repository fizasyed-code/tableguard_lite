import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from tableguard.datasets import episode_files_v2, summarize_metadata, validate_relative_file, fetch_reference, _validate_repo
from tableguard.common import write_json


def fake_info():
    # Test fixture, not a downloaded dataset record.
    return {'codebase_version':'v2.1','robot_type':'fixture_robot','total_episodes':2,'total_frames':4,
            'chunks_size':1000,'fps':30,
            'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features':{'action':{'dtype':'float32','shape':[6]},
                        'observation.images.top':{'dtype':'video','shape':[480,640,3]}}}

class DatasetTests(unittest.TestCase):
    def test_summarizes_action_shape(self):
        self.assertEqual(summarize_metadata(fake_info())['action_shape'],[6])
    def test_resolves_paths_from_metadata(self):
        paths=episode_files_v2(fake_info(),1)
        self.assertEqual(paths[0],'data/chunk-000/episode_000001.parquet')
        self.assertEqual(paths[1],'videos/chunk-000/observation.images.top/episode_000001.mp4')
    def test_negative_episode_rejected(self):
        with self.assertRaises(ValueError): episode_files_v2(fake_info(),-1)
    def test_out_of_range_episode_rejected(self):
        with self.assertRaises(ValueError): episode_files_v2(fake_info(),2)
    def test_unknown_camera_rejected(self):
        with self.assertRaises(ValueError): episode_files_v2(fake_info(),0,'missing_camera')
    def test_v3_refuses_guessing_episode_shards(self):
        info=fake_info();info['codebase_version']='v3.0'
        with self.assertRaises(ValueError): episode_files_v2(info,0)
    def test_traversal_rejected(self):
        with self.assertRaises(ValueError): validate_relative_file('../private.json')
    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError): validate_relative_file('/tmp/x')
    def test_windows_path_rejected(self):
        with self.assertRaises(ValueError): validate_relative_file('C:\\temp\\x')
    def test_invalid_repo_rejected(self):
        with self.assertRaises(ValueError): _validate_repo('https://example.com/x')
    def test_invalid_chunk_size_rejected(self):
        info=fake_info();info['chunks_size']=0
        with self.assertRaises(ValueError): episode_files_v2(info,0)
    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status.json';write_json(p,{'state':'running'});write_json(p,{'state':'stopped'})
            self.assertEqual(json.loads(p.read_text())['state'],'stopped')
            self.assertEqual(len(list(Path(d).iterdir())),1)
    def test_nan_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): write_json(Path(d)/'x.json',{'x':float('nan')})
    def _fake_hub(self):
        """Mock the network so offline tests never claim a public download."""
        info=json.dumps(fake_info()).encode()
        files={'meta/info.json':info,'README.md':b'TEST FIXTURE ONLY',
               'data/chunk-000/episode_000000.parquet':b'FAKE PARQUET FOR DOWNLOAD TEST',
               'videos/chunk-000/observation.images.top/episode_000000.mp4':b'FAKE VIDEO FOR DOWNLOAD TEST'}
        mod=ModuleType('huggingface_hub')
        class API:
            def __init__(self,**kwargs):pass
            def dataset_info(self,**kwargs):
                return SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename=n) for n in files])
        def url(repo,name,**kwargs): return name
        def metadata(url,**kwargs):return SimpleNamespace(size=len(files[url]),commit_hash='a'*40)
        def download(repo_id,filename,local_dir,**kwargs):
            p=Path(local_dir)/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(files[filename]);return str(p)
        mod.HfApi=API;mod.hf_hub_url=url;mod.get_hf_file_metadata=metadata;mod.hf_hub_download=download
        return mod
    def test_mocked_metadata_download_pins_revision(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',{'huggingface_hub':self._fake_hub()}):
            p=fetch_reference('test/reference',output_root=Path(d))
            m=json.loads(p.read_text());self.assertEqual(m['resolved_revision'],'a'*40)
            self.assertEqual(m['status'],'complete');self.assertEqual(len(m['files']),2)
    def test_mocked_single_episode_download(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',{'huggingface_hub':self._fake_hub()}):
            p=fetch_reference('test/reference',include_media=True,output_root=Path(d))
            m=json.loads(p.read_text());self.assertEqual(len(m['files']),4)
            self.assertTrue(all(len(f['sha256'])==64 for f in m['files']))
    def test_mocked_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',{'huggingface_hub':self._fake_hub()}):
            with self.assertRaises(RuntimeError): fetch_reference('test/reference',max_mb=0.00001,output_root=Path(d))
