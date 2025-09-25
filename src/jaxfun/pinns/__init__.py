from .embeddings import (
    Embedding as Embedding,
    FourierEmbs as FourierEmbs,
    PeriodEmbs as PeriodEmbs,
)
from .freeze import freeze_layer as freeze_layer, unfreeze_layer as unfreeze_layer
from .loss import LSQR as LSQR
from .mesh import (
    Annulus as Annulus,
    AnnulusPolar as AnnulusPolar,
    Line as Line,
    Rectangle as Rectangle,
    UnitLine as UnitLine,
    UnitSquare as UnitSquare,
)
from .module import (
    Comp as Comp,
    FlaxFunction as FlaxFunction,
)
from .nnspaces import (
    MLPSpace as MLPSpace,
    MLPVectorSpace as MLPVectorSpace,
    PirateSpace as PirateSpace,
)
from .optimizer import (
    GaussNewton as GaussNewton,
    adam as adam,
    lbfgs as lbfgs,
    run_optimizer as run_optimizer,
    soap as soap,
)

# from .bcs import DirichletBC
