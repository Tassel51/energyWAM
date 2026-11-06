"""Regressions against the probed official RoboCasa archive layouts."""
import json,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import robocasa_download_worker as w
from test_download_worker import make_lerobot_dir,make_ctx,make_entry
from unittest.mock import patch
import errno
class OfficialArchiveRegressions(unittest.TestCase):
 def test_v2_chunks_size_counts_episode_files(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);info=make_lerobot_dir(p,episodes=2,frames=10);info.pop('total_parquet');info.update(codebase_version='v2.1',chunks_size=1000);(p/'info.json').write_text(json.dumps(info))
   self.assertEqual(w.verify_lerobot_dir(p)['parquet'],2)
 def test_asset_readme_alongside_fixtures(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);archive=p/'fixtures.zip'
   with zipfile.ZipFile(archive,'w') as z:z.writestr('README.md','official archive note');z.writestr('fixtures/stand_mixers/item/model.xml','<mujoco/>')
   out=p/'out';w.extract_asset_zip(archive,out,{'fixtures'})
   self.assertTrue((out/'stand_mixers/item/model.xml').is_file());self.assertFalse((out/'README.md').exists())
 def test_complete_partial_needs_no_extra_request(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'archive.part';p.write_bytes(b'data')
   self.assertEqual(w._download_once('http://invalid.invalid',p,4,True,None),(4,4))
 def test_completed_extraction_counts_toward_ceiling(self):
  with tempfile.TemporaryDirectory() as d:
   g=w.SpaceGuard(Path(d),0.01);g.record_extracted(20*1024**2)
   with self.assertRaises(w.BudgetError):g.check_extract(expected_bytes=1)
 def test_quota_error_pauses_queue(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=make_ctx(Path(d));entry=make_entry()
   with patch.object(w,'_process_entry',side_effect=OSError(errno.EDQUOT,'quota exceeded')):
    rec=w.process_entry(entry,ctx)
   self.assertTrue(ctx.stop.is_set());self.assertEqual(rec['state'],'blocked_budget')
if __name__=='__main__':unittest.main(verbosity=2)
