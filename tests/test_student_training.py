import tempfile, unittest
try:
 import torch
 from hex_reconstruction.student_training import Group49Student,balanced_game_split,soft_policy_loss,weighted_mse
 HAS_TORCH=True
except ModuleNotFoundError: HAS_TORCH=False

@unittest.skipUnless(HAS_TORCH,'torch is installed in reconstruction training venv')
class StudentTests(unittest.TestCase):
 def test_default_architecture_is_existing_10x256_baseline(self):
  m=Group49Student();self.assertEqual(sum(p.numel() for p in m.parameters()),11864485)
 def test_shapes_and_deterministic_checkpoint(self):
  torch.manual_seed(7);m=Group49Student(channels=8,blocks=1).eval();x=torch.zeros(2,6,11,11);p,v=m(x);self.assertEqual(tuple(p.shape),(2,121));self.assertEqual(tuple(v.shape),(2,))
  with tempfile.NamedTemporaryFile() as f:
   torch.save(m.state_dict(),f.name); n=Group49Student(channels=8,blocks=1).eval();n.load_state_dict(torch.load(f.name,weights_only=True));self.assertTrue(torch.equal(m(x)[0],n(x)[0]))
 def test_soft_ce_not_argmax_and_masking(self):
  logits=torch.tensor([[2.,0.],[2.,0.]]) ;pi=torch.tensor([[.5,.5],[1.,0.]])
  got=soft_policy_loss(logits,pi,torch.tensor([1.,0.])); expected=-(.5*torch.log_softmax(logits[0],0)).sum();self.assertTrue(torch.allclose(got,expected));self.assertNotAlmostEqual(got.item(),torch.nn.functional.cross_entropy(logits[:1],torch.tensor([0])).item())
 def test_value_perspective_weights(self):
  self.assertAlmostEqual(weighted_mse(torch.tensor([1.,-1.]),torch.tensor([1.,1.]),torch.tensor([1.,0.])).item(),0.)
 def test_game_level_balanced_split_is_reproducible(self):
  games=[(f'g{i:02d}',bool(i%2)) for i in range(64)];a=balanced_game_split(games);b=balanced_game_split(games);self.assertEqual(a,b);self.assertEqual(sum(x=='train' for x in a.values()),48);self.assertEqual(sum(a[f'g{i:02d}']=='train' for i in range(64) if i%2),24)
 def test_generalized_balanced_split_preserves_swap_balance(self):
  games=[(f'g{i:04d}',bool(i%2)) for i in range(1920)]; split=balanced_game_split(games,dataset_id='scaled',train_per_swap=768); by_id=dict(games)
  self.assertEqual(sum(value=='train' for value in split.values()),1536);self.assertEqual(sum(value=='validation' for value in split.values()),384)
  self.assertEqual(sum(by_id[gid] for gid,value in split.items() if value=='train'),768);self.assertEqual(sum(by_id[gid] for gid,value in split.items() if value=='validation'),192)
