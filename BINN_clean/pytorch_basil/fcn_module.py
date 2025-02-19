
from torch import nn
from torch import Tensor
from typing import Optional, Union


def get_activation(activation: Union[str, None]) -> nn.Module:
    """Get PyTorch activation function by string query.

    Args:
        activation (str, None): activation function, one of `relu`, `leakyrelu`,
            `selu`, `sigmoid`, `softplus`, `tanh`, `identity` (aka `linear` or `none`).
            If `None` is passed, `None` is returned.

    Raises:
        ValueError: if activation not found.

    Returns:
        PyTorch activation or None: activation function
    """

    if activation is None:
        return nn.Identity()

    a = activation.lower()

    if a == 'linear' or a == 'none':
        a = 'identity'

    activations = dict(
        relu=nn.ReLU,
        leakyrelu=nn.LeakyReLU,
        selu=nn.SELU,
        sigmoid=nn.Sigmoid,
        softplus=nn.Softplus,
        tanh=nn.Tanh,
        identity=nn.Identity
    )

    if a in activations:
        return activations[a]()
    else:
        choices = ', '.join(list(activations.keys()))
        raise ValueError(
            f'activation `{activation}` not found, chose one of: {choices}.'
        )


class FeedForward(nn.Module):
    """Implements a n-layered feed-forward neural networks.

    Implements a feed-forward model, each layer conisting of a linear,
    dropout and an activation layer. The dropout and actiavation of the
    last layer are optional (see `activation_last` and `dropout_last`).

    Args:
        num_inputs (int): input dimensionality
        num_outputs (int): output dimensionality
        num_hidden (int): number of hidden units
        num_layers (int): number of hidden fully-connected layers
        dropout (float): dropout applied after each layer, in range [0, 1)
        activation (str): activation function, defaults to 'relu'.
        activation_last (str, optional): if not `None`, this
            activation is applied after the last layer. Defaults to None.
        dropout_last (bool, optional): If `True`, the dropout is also
            applied after last layer. Defaults to False.

    """
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            activation: str = 'relu',
            activation_last: Optional[str] = None,
            dropout_last: bool = False) -> None:

        super().__init__()

        activation_fn = get_activation(activation)
        activation_last_fn = get_activation(activation_last)

        in_sizes = [num_inputs] + [num_hidden] * (num_layers)
        out_sizes = [num_hidden] * (num_layers) + [num_outputs]

        layers = {}
        is_last = False
        for idx, (ni, no) in enumerate([(ni, no) for ni, no in
                                        zip(in_sizes, out_sizes)]):

            layer = nn.Linear(ni, no)

            if idx == num_layers:
                is_last = True
            layers.update({f'linear{idx:02d}': layer})

            if not is_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})
                layers.update({f'activation{idx:02d}': activation_fn})

            if is_last and dropout_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})

            if is_last and activation_last is not None:
                layers.update({f'activation{idx:02d}': activation_last_fn})

        self.model = nn.Sequential()

        for k, v in layers.items():
            self.model.add_module(k, v)

    def forward(self, x: Tensor) -> Tensor:
        """Model forward call.

        Args:
            x (Tensor): the sequencial tensor with shape (batch_size, sequence_length, features).

        Returns:
            Tensor: the output tensor.
        """
        return self.model(x)
