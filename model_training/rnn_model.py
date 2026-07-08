import torch 
from torch import nn

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
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
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


class ResidualConv1d(nn.Module):
    """Residual dilated convolution for tracking short and long range
    temporal patterns."""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        # Keeps the original signal (residual connection)
        # While also adding new learned correction
        return x + self.net(x)


class GLUConv1d(nn.Module):
    """Gated linear unit layer that determines how strongly the learned
    features should pass through."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=2 * channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class TemporalBlock(nn.Module):
    """Residual dilation convolution block with increasing dilation.
    GLU is implemented at the end to determine how much of each feature
    should be passed forward."""
    def __init__(self, channels=512, kernel_size=3, dilation1=1, dilation2=2):
        super().__init__()

        self.res1 = ResidualConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation1,
        )

        self.res2 = ResidualConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation2,
        )

        self.glu = GLUConv1d(
            channels=channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.glu(x)
        return x


class TCNDecoder(nn.Module):
    """
    Defines the TCN decoder.

    This class combines day-specific input layers, a TCN, and an output classification layer.
    """
    def __init__(self,
                 neural_dim,
                 n_days,
                 n_classes,
                 input_dropout = 0.4,
                 kernel_size=3,
                 tcn_num_blocks = 6,
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
        self.kernel_size = kernel_size
        self.tcn_num_blocks = tcn_num_blocks

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

        self.residual_blocks = nn.ModuleList()

        for k in range(tcn_num_blocks):
            # Exponentially increasing dilations expand the receptive field
            # while keeping the per-block parameter count fixed.
            dilation1 = 2 ** k
            dilation2 = 2 ** (k + 1)
            self.residual_blocks.append(
                TemporalBlock(
                    channels=neural_dim,
                    kernel_size=kernel_size,
                    dilation1=dilation1,
                    dilation2=dilation2,
                )
            )

        self.out = nn.Linear(self.neural_dim, self.n_classes)
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

        # Conv1d expects [batch, channels, time].
        x = x.permute(0, 2, 1)

        for block in self.residual_blocks:
            x = block(x)

        # Convert back to [batch, time, channels] for the classifier head.
        x = x.permute(0, 2, 1)
        logits = self.out(x)

        if return_state:
            return logits, None

        return logits

    def compute_output_lengths(self, input_lengths):
        return input_lengths.to(torch.int32)
