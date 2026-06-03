
import joblib


def load_scalers(scaler_dir):
    scaler_C11 = joblib.load(scaler_dir / "scaler_C11")
    scaler_C12 = joblib.load(scaler_dir / "scaler_C12")
    scaler_C66 = joblib.load(scaler_dir / "scaler_C66")

    return scaler_C11, scaler_C12, scaler_C66




def load_neural_field(nf_config):
    """
    TODO: Load pretrained neural field model from checkpoint specified in nf_config.
    """
    ...