from typing import Iterable
from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import CoordSys

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00D7"

class TensorProductSpace:
    def __init__(
        self,
        spaces: list[BaseSpace],
        coordinates: CoordSys = None,
        name: str = None,
    ) -> None:
        from jaxfun.arguments import CartCoordSys
        self.spaces = spaces
        self.name = name
        self.system = CartCoordSys[len(spaces)] if coordinates is None else coordinates
        self.tensorname = tensor_product_symbol.join([b.name for b in spaces])

    def __len__(self) -> int:
        return len(self.spaces)

    def __iter__(self) -> Iterable[BaseSpace]:
        return iter(self.spaces)
    
    def __getitem__(self, i: int) -> BaseSpace:
        return self.spaces[i]
    
    @property
    def rank(self):
        return 0


class VectorTensorProductSpace:
    def __init__(
        self,
        tensorspace: TensorProductSpace | tuple[TensorProductSpace],
        name: str = None,
    ) -> None:
        if not isinstance(tensorspace, tuple):
            assert isinstance(tensorspace, TensorProductSpace)
            tensorspace = (tensorspace,) * len(tensorspace)
        self.tensorspaces = tensorspace
        self.system = self.tensorspaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.tensorspaces])

    def __len__(self) -> int:
        return len(self.tensorspaces)

    def __iter__(self) -> Iterable[TensorProductSpace]:
        return iter(self.tensorspaces)

    def __getitem__(self, i: int) -> TensorProductSpace:
        return self.tensorspaces[i]

    @property
    def rank(self):
        return 1
