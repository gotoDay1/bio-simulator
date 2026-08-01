from biosim.models.base import GrowthModel, OxygenModel, ProductModel, SubstrateModel
from biosim.models.growth import GompertzGrowth, LogisticGrowth, MonodGrowth
from biosim.models.oxygen import OxygenDemandOnly, OxygenWithKLa
from biosim.models.product import LuedekingPiretProduct, NoProduct
from biosim.models.substrate import YieldMaintenanceSubstrate

GROWTH_MODELS: dict[str, type[GrowthModel]] = {
    "monod": MonodGrowth,
    "logistic": LogisticGrowth,
    "gompertz": GompertzGrowth,
}

PRODUCT_MODELS: dict[str, type[ProductModel]] = {
    "luedeking_piret": LuedekingPiretProduct,
    "none": NoProduct,
}

SUBSTRATE_MODELS: dict[str, type[SubstrateModel]] = {
    "yield_maintenance": YieldMaintenanceSubstrate,
}

OXYGEN_MODELS: dict[str, type[OxygenModel]] = {
    "demand_only": OxygenDemandOnly,
    "kla_supply": OxygenWithKLa,
}
