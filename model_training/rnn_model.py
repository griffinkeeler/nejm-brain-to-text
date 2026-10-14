import torch
from torch import nn
import torch.nn.functional as F

def temporal_patches(x, patch_size, patch_stride):
    """Flatten time-by-feature patches, including on older MPS backends."""
    if x.device.type == 'mps':
        # PyTorch 2.5 MPS has no unfold backward. index_select preserves the
        # same ordering and accumulates gradients for overlapping patches.
        starts = torch.arange(0, x.shape[1] - patch_size + 1,
                              patch_stride, device=x.device)
        offsets = torch.arange(patch_size, device=x.device)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)
        return x.index_select(1, indices).reshape(
            x.shape[0], starts.numel(), patch_size * x.shape[2])
    patches = x.unfold(1, patch_size, patch_stride)
    return patches.permute(0, 1, 3, 2).reshape(
        x.shape[0], patches.shape[1], patch_size * x.shape[2])


class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 frame_embed_dim = None,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        frame_embed_dim(int)  - optional shared frame width; patches project to n_units
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        if frame_embed_dim is not None and (
                isinstance(frame_embed_dim, bool) or
                not isinstance(frame_embed_dim, int) or frame_embed_dim <= 0):
            raise ValueError('frame_embed_dim must be a positive integer or None')
        self.frame_embed_dim = frame_embed_dim

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.frame_embed = None
        self.patch_embed = None
        if frame_embed_dim is not None:
            self.frame_embed = nn.Sequential(
                nn.LayerNorm(self.neural_dim),
                nn.Linear(self.neural_dim, frame_embed_dim),
                nn.GELU(),
            )
            patch_dim = frame_embed_dim * max(self.patch_size, 1)
            self.patch_embed = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, self.n_units),
                nn.LayerNorm(self.n_units),
            )
            nn.init.xavier_uniform_(self.frame_embed[1].weight)
            nn.init.xavier_uniform_(self.patch_embed[1].weight)
            self.input_size = self.n_units

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.frame_embed is not None:
            x = self.frame_embed(x)
        if self.patch_size > 0: 
  
            x = temporal_patches(x, self.patch_size, self.patch_stride)

        if self.patch_embed is not None:
            x = self.patch_embed(x)

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits

    def compute_output_lengths(self, input_lengths):
        if self.patch_size > 0:
            return ((input_lengths - self.patch_size) / self.patch_stride + 1).to(torch.int32)
        return input_lengths.to(torch.int32)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for channel-first temporal tensors."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PortableConv1d(nn.Conv1d):
    """Older MPS convolution backward requires contiguous output gradients."""
    def forward(self, x):
        output = super().forward(x)
        if output.device.type == 'mps' and output.requires_grad:
            output.register_hook(lambda grad: grad.contiguous())
        return output


class CausalConv1d(nn.Module):
    """1D convolution with left padding so each timestep only sees the past."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = PortableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CompactTemporalBlock(nn.Module):
    """Compact causal TCN block with depthwise temporal mixing."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0, expansion_factor=2,
                 use_glu=False):
        super().__init__()

        expanded_channels = channels * expansion_factor

        self.norm1 = ChannelLayerNorm(channels)
        self.depthwise = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise_in = PortableConv1d(
            channels, expanded_channels * (2 if use_glu else 1), kernel_size=1)
        self.pointwise_out = PortableConv1d(expanded_channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.expansion_activation = nn.GLU(dim=1) if use_glu else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = ChannelLayerNorm(channels)

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise_in(x)
        x = self.expansion_activation(x)
        x = self.dropout(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)

        return self.norm2(residual + x)


class TCNDecoder(nn.Module):
    """
    Defines the TCN decoder.

    This class combines day-specific input layers, a TCN, and an output classification layer.
    """
    def __init__(self,
                 neural_dim,
                 n_days,
                 n_classes,
                 input_dropout=0.4,
                 patch_size=5,
                 patch_stride=5,
                 hidden_dim=384,
                 kernel_size=3,
                 tcn_num_blocks=5,
                 tcn_dropout=0.35,
                 expansion_factor=2,
                 frame_embed_dim=None,
                 use_glu=False,
                 residual_fusion=False,
                 ):
        """
        Args:
            neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
            n_days      (int)      - number of days in the dataset
            n_classes   (int)      - number of output classes
            input_dropout (float)  - percentage of input units to dropout during training
        """
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.tcn_num_blocks = tcn_num_blocks
        self.tcn_dropout = tcn_dropout
        self.expansion_factor = expansion_factor
        self.use_glu = use_glu
        if residual_fusion and frame_embed_dim is None:
            raise ValueError('residual_fusion requires frame_embed_dim')
        self.residual_fusion = residual_fusion
        if frame_embed_dim is not None and (
                isinstance(frame_embed_dim, bool) or
                not isinstance(frame_embed_dim, int) or frame_embed_dim <= 0):
            raise ValueError('frame_embed_dim must be a positive integer or None')
        self.frame_embed_dim = frame_embed_dim

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign()  # basically a shallower tanh

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.frame_embed = None
        patch_feature_dim = self.neural_dim
        if frame_embed_dim is not None:
            self.frame_embed = nn.Sequential(
                nn.LayerNorm(self.neural_dim),
                nn.Linear(self.neural_dim, frame_embed_dim),
                nn.GELU(),
            )
            nn.init.xavier_uniform_(self.frame_embed[1].weight)
            patch_feature_dim = frame_embed_dim

        patch_dim = patch_feature_dim * self.patch_size
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        nn.init.xavier_uniform_(self.patch_embed[1].weight)

        self.raw_patch_embed = None
        self.output_projection = None
        if residual_fusion:
            raw_patch_dim = self.neural_dim * self.patch_size
            self.raw_patch_embed = nn.Sequential(
                nn.LayerNorm(raw_patch_dim),
                nn.Linear(raw_patch_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            nn.init.xavier_uniform_(self.raw_patch_embed[1].weight)
            self.output_projection = nn.Sequential(
                PortableConv1d(2 * self.hidden_dim, 2 * self.hidden_dim, kernel_size=1),
                nn.GELU(),
                PortableConv1d(2 * self.hidden_dim, self.hidden_dim, kernel_size=1),
            )

        self.residual_blocks = nn.ModuleList()

        for k in range(tcn_num_blocks):
            dilation = 2 ** k
            self.residual_blocks.append(
                CompactTemporalBlock(
                    channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                    expansion_factor=expansion_factor,
                    use_glu=use_glu,
                )
            )

        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states = None, return_state = False):
        """
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x.
        """

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        raw_embedding = None
        if self.residual_fusion:
            raw_embedding = self.raw_patch_embed(
                temporal_patches(x, self.patch_size, self.patch_stride))
        if self.frame_embed is not None:
            x = self.frame_embed(x)
        x = temporal_patches(x, self.patch_size, self.patch_stride)
        x = self.patch_embed(x)

        # Conv1d expects [batch, channels, time].
        if self.residual_fusion:
            fused = self.output_projection(
                torch.cat([raw_embedding, x], dim=-1).transpose(1, 2))
            x = raw_embedding.transpose(1, 2) + fused
        else:
            x = x.permute(0, 2, 1)

        for block in self.residual_blocks:
            x = block(x)

        # Convert back to [batch, time, channels] for the classifier head.
        x = x.permute(0, 2, 1)
        x = self.out_norm(x)
        logits = self.out(x)

        if return_state:
            return logits, None

        return logits

    def compute_output_lengths(self, input_lengths):
        return ((input_lengths - self.patch_size) / self.patch_stride + 1).to(torch.int32)
