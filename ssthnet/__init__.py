"""Public model factory for the single-shared-expert SSTHNet."""
from ._architecture import SSTHMoEAblation


def make_model():
    """Construct the final architecture; weights must be loaded separately."""
    model = SSTHMoEAblation("single_expert")
    if model.router is not None or len(model.private_experts):
        raise RuntimeError("The final SSTHNet must use one shared expert only")
    return model
