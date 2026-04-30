import jax.numpy as jnp


def filter_dict_to_tuples(d):
    """
    Recursively process a nested dictionary and convert innermost dictionaries
    with integer keys to tuples.
    """
    if not isinstance(d, dict):
        return d

    # First, recursively process all values
    processed = {k: filter_dict_to_tuples(v) for k, v in d.items()}

    # Then check if this dictionary should be converted to a tuple
    # Only convert if all keys are integers AND all values are not dictionaries
    if all(
        isinstance(k, int) and not isinstance(v, dict) for k, v in processed.items()
    ):
        # Convert to tuple, sorted by keys
        return tuple(processed[k] for k in sorted(processed.keys()))

    return processed


d = {
    "hidden": {
        0: {
            "bias": {0: jnp.array([1, 2, 3])},
            "kernel": {0: jnp.array([1, 2, 3]), 1: jnp.array([4, 5, 6])},
        }
    },
    "linear_in": {
        "bias": {0: jnp.array([1, 2, 3])},
        "kernel": {0: jnp.array([1, 2]), 1: jnp.array([4, 5, 6])},
    },
    "linear_out": {
        "bias": {0: jnp.array([1, 2])},
        "kernel": {0: jnp.array([1, 2, 3]), 1: jnp.array([1, 2])},
    },
}

# Apply the filter
d_filtered = filter_dict_to_tuples(d)
print("Original d['hidden'][0]['bias']:", d["hidden"][0]["bias"])
print("Filtered d['hidden'][0]['bias']:", d_filtered["hidden"][0]["bias"])
print("\nOriginal d['hidden'][0]['kernel']:", d["hidden"][0]["kernel"])
print("Filtered d['hidden'][0]['kernel']:", d_filtered["hidden"][0]["kernel"])
