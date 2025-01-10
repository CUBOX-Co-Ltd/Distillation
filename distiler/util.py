# For detection dataset
def preprocess_batch(batch, device, precision):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch["img"] = (batch["img"].to(device, non_blocking=True).float() / 255).to(precision)
    return batch