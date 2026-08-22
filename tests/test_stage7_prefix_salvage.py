import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('salvage',Path(__file__).resolve().parents[1]/'scripts'/'salvage_stage7_prefixes_v1.py')
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
def game(n=10,w='B'): return {'game_id':77,'winner':w,'game_length':n,'moves':[0]*n,'samples':[{'ply':i,'selected_move':0,'side_to_move':'B' if i%2==0 else 'W','root_visits':[128]+[0]*120} for i in range(n)],'search_budget':128}
def cert(ply,owner='B',**kw): return {'certificate_ply':ply,'certificate_owner':owner,'bridge_count':2,'validated':True,'realizer_supported':True,'simultaneous':False,**kw}
class PrefixSalvageTests(unittest.TestCase):
 def source(self):
  directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);p=Path(directory.name)/'game.json';p.write_text('{}');return p
 def test_no_certificate_keeps_literal_game(self):
  r=M.classify(game(),self.source(),{'certificate_ply':None});self.assertEqual((r['retained_phase_a_rows'],r['effective_winner']),(10,'B'))
 def test_producing_move_is_retained_and_flip_relabels(self):
  r=M.classify(game(10,'W'),self.source(),cert(3,'B'));self.assertEqual(r['retained_phase_a_rows'],3);self.assertTrue(r['winner_changed']);self.assertEqual(r['effective_winner'],'B')
 def test_same_move_literal_terminal_keeps_all(self):
  r=M.classify(game(10,'W'),self.source(),cert(10,'W'));self.assertEqual(r['retained_phase_a_rows'],10);self.assertFalse(r['winner_changed'])
 def test_invalid_or_simultaneous_quarantines(self):
  self.assertEqual(M.classify(game(),self.source(),cert(3,'B',validated=False))['status'],'quarantined')
  self.assertEqual(M.classify(game(),self.source(),cert(3,'B',simultaneous=True))['status'],'quarantined')
 def test_nonmoving_owner_quarantines(self):
  self.assertEqual(M.classify(game(),self.source(),cert(3,'W'))['reason'],'certificate_first_seen_for_nonmoving_side')
 def test_colour_transpose_preserves_prefix_contract(self):
  black=M.classify(game(10,'W'),self.source(),cert(3,'B'))
  white=M.classify(game(10,'B'),self.source(),cert(4,'W'))
  self.assertEqual(black['retained_phase_a_rows'],white['retained_phase_a_rows']-1)
  self.assertTrue(black['winner_changed'] and white['winner_changed'])
if __name__=='__main__': unittest.main()
