"""Windows input dimension adapter; preserves frozen 3x128 GAT and temporal branches."""
from .tgnn_components import ComponentTGNN
from .temporal_cnn import CNNHybrid, CausalTemporalCNN

ARMS=('GNN_H','TGNN_Q_A_C')
class WindowsHybrid(ComponentTGNN):
 def __init__(self,input_dim):
  super().__init__(input_dim,('T','H','Q','A'))
  self.cnn=CausalTemporalCNN()
  self.factors=frozenset(('T','H','Q','A','C'))
 encode=CNNHybrid.encode

def make_model(arm,input_dim=72):
 if arm=='GNN_H': return ComponentTGNN(input_dim,('H',))
 if arm=='TGNN_Q_A_C': return WindowsHybrid(input_dim)
 raise ValueError(arm)
