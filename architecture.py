import torch
import torch.nn as nn
import torch.nn.functional as F

class DockingModel6K(nn.Module):
    """
    Enhanced neural network model for docking with Monte Carlo Dropout for binary classification.
    """

    def __init__(self, input_size, p=0.5):
        """
        Initialize the DockingModel with dropout for binary classification.

        :param input_size: Number of input features.
        :param p: Dropout probability.
        """
        super(DockingModel6K, self).__init__()

        # Input block
        self.input_layer = nn.Linear(input_size, 6000)  # Increased input layer size
        self.bn1 = nn.BatchNorm1d(6000)

        # Hidden layers with increased depth and capacity
        self.fc1 = nn.Linear(6000, 6000)
        self.bn2 = nn.BatchNorm1d(6000)
        self.fc2 = nn.Linear(6000, 4000)
        self.bn3 = nn.BatchNorm1d(4000)
        self.fc3 = nn.Linear(4000, 3000)
        self.bn4 = nn.BatchNorm1d(3000)
        self.fc4 = nn.Linear(3000, 1500)
        self.bn5 = nn.BatchNorm1d(1500)
        self.fc5 = nn.Linear(1500, 512)
        self.bn6 = nn.BatchNorm1d(512)

        # Additional hidden layer
        self.fc6 = nn.Linear(512, 128)
        self.bn7 = nn.BatchNorm1d(128)

        # Output layer
        self.output_layer = nn.Linear(128, 2)

        # Dropout and residuals
        self.dropout = nn.Dropout(p)

    def get_embedding_dim(self):
        return 128  # Adjusted for the last hidden layer size

    def forward(self, x, last=False):
        """
        Forward pass of the enhanced model with additional layers and normalization.

        :param x: Input tensor.
        :param last: Whether to return intermediate embeddings.
        :return: Logits for binary classification (and optionally embeddings).
        """
        # Input block
        x = F.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)

        # Hidden layers with residual connections
        x1 = F.relu(self.bn2(self.fc1(x)))
        x = x + self.dropout(x1)  # Residual connection

        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        x = F.relu(self.bn5(self.fc4(x)))
        x = F.relu(self.bn6(self.fc5(x)))

        # Additional hidden layer
        x = F.relu(self.bn7(self.fc6(x)))
        dropout_x = self.dropout(x)

        # Output layer
        x = self.output_layer(dropout_x)
        if last:
            return x, dropout_x
        else:
            return x

    def mc_dropout_forward(self, x, n_iter=10):
        """
        Perform Monte Carlo Dropout forward passes for binary classification.

        :param x: Input tensor.
        :param n_iter: Number of Monte Carlo iterations.
        :return: Mean and standard deviation of the predictions.
        """
        self.train()  # Set the model to training mode to apply dropout
        predictions = torch.stack([self.forward(x) for _ in range(n_iter)], dim=0)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        self.eval()  # Set the model back to evaluation mode
        return mean_prediction, std_prediction


class DockingModelTx(nn.Module):
    """
    Advanced docking model with state-of-the-art architectural enhancements.
    """

    def __init__(self, input_size, p=0.5):
        """
        Initialize the DockingModel with advanced features.

        :param input_size: Number of input features.
        :param p: Dropout probability.
        """
        super(DockingModelTx, self).__init__()

        # Input projection
        self.input_layer = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        # Transformer-inspired encoder block
        self.encoder_block1 = self._create_transformer_block(2048, 8, 0.1)
        self.encoder_block2 = self._create_transformer_block(2048, 8, 0.1)

        # Highway network
        self.highway_layer = nn.Linear(2048, 2048)
        self.highway_gate = nn.Linear(2048, 2048)

        # Hidden layers
        self.fc1 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)

        # Output layer
        self.output_layer = nn.Linear(128, 2)

        # Dropout
        self.dropout = nn.Dropout(p)

    def _create_transformer_block(self, dim, num_heads, dropout_prob):
        """
        Create a transformer-inspired encoder block with self-attention and feedforward layers.

        :param dim: Dimensionality of the input.
        :param num_heads: Number of attention heads.
        :param dropout_prob: Dropout probability.
        :return: nn.Sequential representing the transformer block.
        """
        return nn.Sequential(
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout_prob, batch_first=True),
            nn.LayerNorm(dim),
            nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(4 * dim, dim),
                nn.Dropout(dropout_prob)
            ),
            nn.LayerNorm(dim),
        )

    def forward(self, x, last=False):
        """
        Forward pass of the advanced model.

        :param x: Input tensor.
        :param last: Whether to return intermediate embeddings.
        :return: Logits for binary classification (and optionally embeddings).
        """
        # Input projection
        x = F.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)

        # Transformer-inspired encoder blocks
        x, _ = self.encoder_block1[0](x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))  # Self-attention
        x = x.squeeze(1)
        x = self.encoder_block1[1](x + self.encoder_block1[2](x))

        x, _ = self.encoder_block2[0](x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))  # Self-attention
        x = x.squeeze(1)
        x = self.encoder_block2[1](x + self.encoder_block2[2](x))

        # Highway network
        highway_out = F.relu(self.highway_layer(x))
        highway_gate = torch.sigmoid(self.highway_gate(x))
        x = highway_gate * highway_out + (1 - highway_gate) * x  # Gate mechanism

        # Hidden layers
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.bn4(self.fc3(x)))

        # Dropout and output
        dropout_x = self.dropout(x)
        x = self.output_layer(dropout_x)
        if last:
            return x, dropout_x
        else:
            return x

    def mc_dropout_forward(self, x, n_iter=10):
        """
        Perform Monte Carlo Dropout forward passes for uncertainty estimation.

        :param x: Input tensor.
        :param n_iter: Number of Monte Carlo iterations.
        :return: Mean and standard deviation of predictions.
        """
        self.train()  # Ensure dropout is active
        predictions = torch.stack([self.forward(x) for _ in range(n_iter)], dim=0)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        self.eval()  # Switch back to evaluation mode
        return mean_prediction, std_prediction

class DockingModel3K(nn.Module):
    """
    Neural network model for docking with Monte Carlo Dropout for binary classification.
    """

    def __init__(self, input_size, p=0.5):
        """
        Initialize the DockingModel with dropout for binary classification.

        :param input_size: Number of input features.
        :param p: Dropout probability.
        """
        super(DockingModel3K, self).__init__()
        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 512)  # Change output layer to have 2 output nodes for binary classification
        self.fc4 = nn.Linear(512, 32)  # Change output layer to have 2 output nodes for binary classification
        self.fc5 = nn.Linear(32, 2)  # Change output layer to have 2 output nodes for binary classification

        self.dropout = nn.Dropout(p)

    def get_embedding_dim(self):
        return 32 #2000 # to be changed maybe

    def forward(self, x, last=False):
        """
        Forward pass of the model with dropout for binary classification.

        :param x: Input tensor.
        :return: Output tensor with logits for binary classification.
        """
        # x = x.to(device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(x))))))
        dropout_x = self.dropout(x)  # Apply dropout
        x = self.fc5(dropout_x)
        if last:
            return x, dropout_x
        else:
            return x

    def mc_dropout_forward(self, x, n_iter=10):
        """
        Perform Monte Carlo Dropout forward passes for binary classification.

        :param x: Input tensor.
        :param n_iter: Number of Monte Carlo iterations.
        :return: Mean and standard deviation of the predictions.
        """
        self.train()  # Set the model to training mode to apply dropout
        predictions = torch.stack([self.forward(x) for _ in range(n_iter)], dim=0)
        # print('mcdropout fwd predictions ', predictions.shape, predictions)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        self.eval()  # Set the model back to evaluation mode
        return mean_prediction, std_prediction
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL')
try:
	import dgl
	from packaging import version
	from layers import GraphConvolution
	from layers import GraphIsomorphism
	from layers import GraphIsomorphismEdge
	from layers import GraphAttention
	from layers import PMALayer
except Exception as e:
	print('DGL absent')

class MyModel(nn.Module):
	def __init__(
			self, 
			model_type,
			num_layers=4, 
			hidden_dim=64,
			num_heads=4, # Only used for GAT
			dropout_prob=0.2,
			bias_mlp=True,
			out_dim=1,
			readout='sum',
			act=F.relu,
			initial_node_dim=58,
			initial_edge_dim=6,
			apply_sigmoid=False,
			multiply_num_pma=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.embedding_node = nn.Linear(initial_node_dim, hidden_dim, bias=False)
		self.embedding_edge = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
		self.readout = readout

		self.mp_layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			mp_layer = None
			if model_type == 'gcn':
				mp_layer = GraphConvolution(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
				)
			elif model_type == 'gin':
				mp_layer = GraphIsomorphism(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gine':
				mp_layer = GraphIsomorphismEdge(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gat':
				mp_layer = GraphAttention(
					hidden_dim=hidden_dim,
					num_heads=num_heads,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			else:
				raise ValueError('Invalid model type: you should choose model type in [gcn, gin, gin, gat, ggnn]')
			self.mp_layers.append(mp_layer)

		if self.readout == 'pma':
			self.pma = PMALayer(
				k=1,
				hidden_dim=hidden_dim,
				num_heads=num_heads,
				multiply_num_pma=multiply_num_pma
			)

		self.linear_out = nn.Linear(hidden_dim, out_dim, bias=True)

		self.apply_sigmoid = apply_sigmoid
		if self.apply_sigmoid:
			self.sigmoid = F.sigmoid
	

	def forward(
			self, 
			graph,
			feat,
			eweight=None,
			training=False,
		):
		h = self.embedding_node(feat.float())
		if eweight==None:
			eweight=graph.edata['e_ij']
		e_ij = self.embedding_edge(eweight.float())
		graph.ndata['h'] = h
		graph.edata['e_ij'] = e_ij

# Update the node features
		for i in range(self.num_layers):
			graph = self.mp_layers[i](
				graph=graph,
				training=training
				)

# Aggregate the node features and apply the last linear layer to compute the logit
		alpha = None
		if self.readout in ['sum', 'mean', 'max']:
			out = dgl.readout_nodes(graph, 'h', op=self.readout)
		elif self.readout == 'pma':
			out, alpha = self.pma(graph)
		out = self.linear_out(out)

		if self.apply_sigmoid:
			out = self.sigmoid(out)

		return out, alpha
try:      
    from transformers import AutoModel, AutoTokenizer      
except Exception as e:
      print('Transformers not found')
      
class MoLFormer(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(MoLFormer, self).__init__()
        smiles_model = "ibm/MoLFormer-XL-both-10pct"
        # Load the pre-trained SMILES tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        # Add additional layers for binary classification
        self.fc1 = nn.Linear(self.base_model.config.hidden_size, 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 2)  # Change the output layer to have 2 output nodes for binary classification

    def freeze_base_model(self, num_layers_to_freeze):
        # Freeze the entire base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Ensure the final layers are trainable
        for param in self.fc1.parameters():
            param.requires_grad = True

        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Forward pass through the base SMILES model
        outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Apply mean pooling over the token embeddings
        pooled_output = torch.mean(outputs, dim=1)

        # Pass the pooled output through additional layers for binary classification
        x = torch.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
        
class EnhancedMoLFormer(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(EnhancedMoLFormer, self).__init__()
        smiles_model = "ibm/MoLFormer-XL-both-10pct"
        # Load the pre-trained SMILES tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        # Define hidden dimensions for the enhanced model
        hidden_dim = self.base_model.config.hidden_size
        intermediate_dim = 3 * hidden_dim

        # Attention pooling mechanism
        self.attention_layer = nn.Linear(hidden_dim, 1)

        # Additional layers for binary classification
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.highway_gate = nn.Linear(intermediate_dim, intermediate_dim)
        self.highway_layer = nn.Linear(intermediate_dim, intermediate_dim)

        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(hidden_dim, 2)  # Output layer for binary classification

    def freeze_base_model(self, num_layers_to_freeze):
        # Freeze the entire base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Ensure the additional layers are trainable
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Forward pass through the base SMILES model
        outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Attention pooling mechanism
        attention_weights = torch.softmax(self.attention_layer(outputs), dim=1)
        pooled_output = torch.sum(attention_weights * outputs, dim=1)

        # Pass the pooled output through the additional layers
        x = self.norm1(F.gelu(self.fc1(pooled_output)))

        # Highway network
        highway_out = F.gelu(self.highway_layer(x))
        highway_gate = torch.sigmoid(self.highway_gate(x))
        x = highway_gate * highway_out + (1 - highway_gate) * x

        x = self.norm2(F.gelu(self.fc2(x)))
        x = self.dropout(x)

        # Output layer
        logits = self.fc3(x)
        return logits


class AdvancedMoLFormer(nn.Module):
    def __init__(self, dropout_rate=0.1, num_dense_layers=3):
        super(AdvancedMoLFormer, self).__init__()
        smiles_model = "ibm/MoLFormer-XL-both-10pct"
        
        # Load the pre-trained SMILES tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

        # Model dimensionality
        hidden_dim = self.base_model.config.hidden_size
        intermediate_dim = 4 * hidden_dim

        # Attention pooling mechanism
        self.attention_layer = nn.Linear(hidden_dim, 1)

        # Dense connections (DenseNet-inspired)
        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_dim + i * hidden_dim, hidden_dim) for i in range(3)
        ])

        concatenated_size = self.base_model.config.hidden_size * (1 + num_dense_layers)
        self.project_dense = nn.Linear(concatenated_size, self.base_model.config.hidden_size)

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.base_model.config.hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
            self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm1 = nn.LayerNorm(intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 2)  # Binary classification

    def _create_residual_block(self, input_dim, output_dim, dropout_rate):
        """Create a single residual block with layer normalization and dropout."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.PReLU(),
        )

    def freeze_base_model(self, num_layers_to_freeze):
        """Freeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Base model output
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

        # Attention pooling
        attention_weights = torch.softmax(self.attention_layer(base_outputs), dim=1)
        pooled_output = torch.sum(attention_weights * base_outputs, dim=1)

        # Dense connections
        dense_out = pooled_output
        for dense_layer in self.dense_layers:
            dense_out = torch.cat([dense_out, dense_layer(dense_out)], dim=1)

        # Projection to match MultiheadAttention embed_dim
        dense_out = self.project_dense(dense_out)

        # Multi-head attention
        attention_out, _ = self.multihead_attention(
            dense_out.unsqueeze(1),  # Query
            dense_out.unsqueeze(1),  # Key
            dense_out.unsqueeze(1)   # Value
        )

        attention_out = attention_out.squeeze(1)

        # Residual blocks
        x = self.residual_blocks(attention_out)

        # Fully connected layers
        x = self.norm1(torch.relu(self.fc1(x)))
        x = self.norm2(torch.relu(self.fc2(x)))

        # Dropout and output
        x = self.dropout(x)
        logits = self.output_layer(x)
        return logits


# class AdvancedMoLFormer(nn.Module):
#     def __init__(self, dropout_rate=0.1, num_dense_layers=6):  # Increased layers
#         super(AdvancedMoLFormer, self).__init__()
#         smiles_model = "ibm/MoLFormer-XL-both-10pct"
        
#         # Load pre-trained tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
#         self.base_model = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)

#         # Increased model dimensionality
#         hidden_dim = self.base_model.config.hidden_size 
#         intermediate_dim = 6 * hidden_dim  # Increased intermediate size

#         # Attention pooling
#         self.attention_layer = nn.Linear(hidden_dim, 1)

#         # Increased number of Dense connections (DenseNet-inspired)
#         self.dense_layers = nn.ModuleList([
#             nn.Linear(hidden_dim + i * hidden_dim, hidden_dim) for i in range(num_dense_layers)
#         ])

#         concatenated_size = self.base_model.config.hidden_size * (1 + num_dense_layers)
#         self.project_dense = nn.Linear(concatenated_size, hidden_dim)

#         # Increased multi-head attention size
#         self.multihead_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=16,  # Increased from 8 to 16
#             dropout=dropout_rate,
#             batch_first=True
#         )

#         # More Residual Blocks
#         self.residual_blocks = nn.Sequential(
#             self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
#             self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
#             self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
#             self._create_residual_block(hidden_dim, intermediate_dim, dropout_rate),
#         )

#         # Fully connected layers with increased width
#         self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
#         self.norm1 = nn.LayerNorm(intermediate_dim)
#         self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Extra fully connected layer

#         # Dropout and output
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.output_layer = nn.Linear(hidden_dim, 2)

#     def _create_residual_block(self, input_dim, output_dim, dropout_rate):
#         """Create a single residual block with layer normalization and dropout."""
#         return nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.LayerNorm(output_dim),
#             nn.PReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(output_dim, input_dim),
#             nn.LayerNorm(input_dim),
#             nn.PReLU(),
#         )

#     def forward(self, input_ids, attention_mask):
#         # Base model output
#         base_outputs = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state

#         # Attention pooling
#         attention_weights = torch.softmax(self.attention_layer(base_outputs), dim=1)
#         pooled_output = torch.sum(attention_weights * base_outputs, dim=1)

#         # Dense connections
#         dense_out = pooled_output
#         for dense_layer in self.dense_layers:
#             dense_out = torch.cat([dense_out, dense_layer(dense_out)], dim=1)

#         # Projection to match MultiheadAttention embed_dim
#         dense_out = self.project_dense(dense_out)

#         # Multi-head attention
#         attention_out, _ = self.multihead_attention(
#             dense_out.unsqueeze(1),  # Query
#             dense_out.unsqueeze(1),  # Key
#             dense_out.unsqueeze(1)   # Value
#         )
#         attention_out = attention_out.squeeze(1)

#         # Residual blocks
#         x = self.residual_blocks(attention_out)

#         # Fully connected layers
#         x = self.norm1(torch.relu(self.fc1(x)))
#         x = self.norm2(torch.relu(self.fc2(x)))
#         x = torch.relu(self.fc3(x))  # Additional FC layer

#         # Dropout and output
#         x = self.dropout(x)
#         logits = self.output_layer(x)
#         return logits



class HybridMolTx(nn.Module):
    def __init__(self, input_size, smiles_model="ibm/MoLFormer-XL-both-10pct", dropout_rate=0.1):
        super(HybridMolTx, self).__init__()

        # Component 1: DockingModelTx backbone
        self.docking_backbone = DockingModelTx(input_size, p=dropout_rate)

        # Component 2: MoLFormer backbone
        self.tokenizer = AutoTokenizer.from_pretrained(smiles_model, trust_remote_code=True)
        self.molformer_base = AutoModel.from_pretrained(smiles_model, deterministic_eval=True, trust_remote_code=True)
        self.molformer_fc = nn.Linear(self.molformer_base.config.hidden_size, 256)

        # Attention mechanism to fuse features
        self.attention_layer = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout_rate, batch_first=True)

        # Auxiliary losses for class imbalance
        self.auxiliary_head_docking = nn.Linear(128, 1)
        self.auxiliary_head_molformer = nn.Linear(256, 1)

        # Combined classification layers
        self.fc1 = nn.Linear(256 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(64, 2)  # Binary classification

    def forward(self, docking_input, smiles_input_ids, smiles_attention_mask):
        # DockingModelTx forward pass
        docking_logits, docking_features = self.docking_backbone(docking_input, last=True)

        # MoLFormer forward pass
        molformer_outputs = self.molformer_base(smiles_input_ids, attention_mask=smiles_attention_mask).last_hidden_state
        molformer_pooled = torch.mean(molformer_outputs, dim=1)
        molformer_features = F.relu(self.molformer_fc(molformer_pooled))

        # Auxiliary loss logits
        aux_docking_logits = self.auxiliary_head_docking(docking_features)
        aux_molformer_logits = self.auxiliary_head_molformer(molformer_features)

        # Fuse features using attention
        fused_features, _ = self.attention_layer(
            docking_features.unsqueeze(1), molformer_features.unsqueeze(1), molformer_features.unsqueeze(1)
        )
        fused_features = fused_features.squeeze(1)

        # Concatenate docking and molformer features
        combined_features = torch.cat([fused_features, docking_features], dim=1)

        # Combined classification layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.output_layer(x)

        return logits, aux_docking_logits, aux_molformer_logits
