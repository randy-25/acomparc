"""Checked integer reference extracted from the validated notebook.

No floating-point activation arithmetic, Torch, or LiteRT inference is used here.
"""
import math
import numpy as np

def checked_integer(values, dtype, label):
    values = np.asarray(values, dtype=object)
    limits = np.iinfo(dtype)

    if int(values.min()) < limits.min or int(values.max()) > limits.max:
        raise OverflowError(
            f"{label}: outside {np.dtype(dtype).name}"
        )

    return values.astype(dtype)

def rq_checked(values, multiplier, shift, label):
    """Match XREQUANT: nearest rounding, ties toward +infinity."""
    if not 1 <= shift <= 63:
        raise ValueError(f"{label}: unsupported shift")

    product = checked_integer(
        np.asarray(values, dtype=object) * int(multiplier),
        np.int64,
        label + " product",
    )
    rounded = checked_integer(
        product.astype(object) + (1 << (shift - 1)),
        np.int64,
        label + " rounding addition",
    )

    return rounded >> shift

def integer_layernorm(input_codes, params):
    if (
        input_codes.dtype not in (np.dtype(np.int8), np.dtype(np.int32))
        or input_codes.shape != (1, 32, 128)
    ):
        raise ValueError("LayerNorm expects INT8 or INT32 [1,32,128]")

    q = input_codes.astype(object)

    # Mean represented by the feature sum.
    mean = checked_integer(
        q.sum(axis=-1, keepdims=True), np.int32, "mean"
    )

    difference = checked_integer(
        128 * q - mean.astype(object), np.int64, "difference"
    )
    square_fine = checked_integer(
        difference.astype(object) ** 2, np.int64, "square"
    )

    # Requantization 1: squared differences -> stored INT32.
    square = checked_integer(
        rq_checked(
            square_fine, 1, params["square_shift"], "square conversion"
        ),
        np.int32, "stored square",
    )

    variance = checked_integer(
        square.astype(object).sum(axis=-1, keepdims=True),
        np.int32, "variance",
    )
    variance_eps = checked_integer(
        variance.astype(object) + params["epsilon_q"],
        np.int32, "variance plus epsilon",
    )

    if np.any(variance_eps <= 0):
        raise ValueError("Nonpositive reciprocal-square-root input")

    inv_std = checked_integer(
        [
            math.isqrt(params["rsqrt_numerator"] // int(v))
            for v in variance_eps.flat
        ],
        np.int32, "inverse standard deviation",
    ).reshape(1, 32, 1)

    normalized_fine = checked_integer(
        difference.astype(object) * inv_std.astype(object),
        np.int64, "normalization",
    )

    # Requantization 2: normalized features -> INT32 at scale 2^-16.
    normalized = checked_integer(
        rq_checked(
            normalized_fine, 1,
            params["normalize_shift"], "normalization conversion",
        ),
        np.int32, "stored normalized features",
    )

    affine_product = checked_integer(
        normalized.astype(object) * params["gamma"].astype(object),
        np.int64, "affine product",
    )
    aligned_beta = checked_integer(
        params["beta"].astype(object) * (1 << 16),
        np.int64, "aligned beta",
    )
    affine = checked_integer(
        affine_product.astype(object) + aligned_beta.astype(object),
        np.int64, "affine sum",
    )

    # Requantization 3: affine INT64 at scale 2^-32 -> INT8.
    shifted = rq_checked(
        affine, params["output_multiplier"],
        params["output_shift"], "LN1 output",
    )
    unbounded = checked_integer(
        shifted.astype(object) + params["output_zero_point"],
        np.int64, "output zero point",
    )
    output = np.clip(unbounded, -128, 127).astype(np.int8)

    return {
        "mean": mean,
        "square_fine": square_fine,
        "square": square,
        "variance": variance,
        "variance_eps": variance_eps,
        "inv_std": inv_std,
        "normalized_fine": normalized_fine,
        "normalized": normalized,
        "affine": affine,
        "output": output,
        "saturation_count": int(np.count_nonzero(
            (unbounded < -128) | (unbounded > 127)
        )),
    }

def integer_ln1(input_codes, params):
    if input_codes.dtype != np.int8:
        raise ValueError("LN1 input must be INT8")
    return integer_layernorm(input_codes, params)

def integer_ln2(input_codes, params):
    if input_codes.dtype != np.int32:
        raise ValueError("LN2 input must be INT32")
    return integer_layernorm(input_codes, params)

def integer_projection(input_codes, params):
    output_features, input_features = params["weights"].shape

    if (
        input_codes.dtype != np.int8
        or input_codes.shape != (1, 32, input_features)
    ):
        raise ValueError("Unexpected INT8 projection input")

    centered = (
        input_codes.reshape(32, input_features).astype(np.int64)
        - params["input_zero_point"]
    )

    accumulator = checked_integer(
        centered @ params["weights"].astype(np.int64).T,
        np.int32, "projection dot product",
    )

    if params.get("bias") is not None:
        accumulator = checked_integer(
            accumulator.astype(object) + params["bias"].astype(object),
            np.int32, "projection bias addition",
        )

    converted = rq_checked(
        accumulator, params["multiplier"], params["shift"],
        "projection requantization",
    )
    unbounded = checked_integer(
        converted.astype(object) + params["output_zero_point"],
        np.int64, "projection zero point",
    )

    return {
        "accumulator": accumulator.reshape(1, 32, output_features),
        "output": np.clip(unbounded, -128, 127).astype(np.int8).reshape(
            1, 32, output_features
        ),
        "saturation_count": int(np.count_nonzero(
            (unbounded < -128) | (unbounded > 127)
        )),
    }

def integer_attention_scores(query, key, params):
    for codes in (query, key):
        if codes.dtype != np.int8 or codes.shape != (1, 32, 128):
            raise ValueError("Q/K must be INT8 [1,32,128]")

    # Token-major [32,128] -> heads [16,32,8].
    query_heads = (
        query.reshape(32, 16, 8)
        .transpose(1, 0, 2)
        .astype(np.int64)
    )
    key_heads = (
        key.reshape(32, 16, 8)
        .transpose(1, 0, 2)
        .astype(np.int64)
    )

    query_heads -= params["query_zero_point"]
    key_heads -= params["key_zero_point"]

    # Each head: [32,8] @ [8,32] -> [32,32].
    scores = query_heads @ key_heads.transpose(0, 2, 1)

    return checked_integer(scores, np.int32, "attention scores")

def integer_causal_softmax(scores, params):
    if scores.dtype != np.int32 or scores.shape != (16, 32, 32):
        raise ValueError("Softmax expects INT32 [16,32,32]")

    allowed = np.tri(32, dtype=bool)[None, :, :]

    # Maximum over allowed keys only.
    maximum = np.where(
        allowed, scores, np.iinfo(np.int32).min
    ).max(axis=-1, keepdims=True)

    shifted = checked_integer(
        scores.astype(object) - maximum.astype(object),
        np.int32, "shifted scores",
    )
    shifted = np.where(allowed, shifted, 0).astype(np.int32)
    assert np.all(shifted <= 0)

    # Rounded conversion to coordinates at scale 2^-16.
    coordinates = rq_checked(
        -shifted.astype(object),
        params["coordinate_multiplier"],
        params["coordinate_shift"],
        "softmax coordinate",
    )

    # Exponentials below the existing cutoff are discarded.
    active = allowed & (coordinates < (16 << 16))
    safe_coordinates = np.where(active, coordinates, 0)

    # LUT spacing: 1/256; remaining eight bits interpolate.
    indices = safe_coordinates >> 8
    fractions = safe_coordinates & 255

    table = params["exp_lut"]
    lower = table[indices].astype(np.int64)
    upper = table[indices + 1].astype(np.int64)

    interpolation_product = checked_integer(
        (upper.astype(object) - lower.astype(object))
        * fractions.astype(object),
        np.int64, "exponential interpolation product",
    )
    adjustment = rq_checked(
        interpolation_product, 1, 8, "exponential interpolation"
    )
    interpolated = checked_integer(
        lower.astype(object) + adjustment.astype(object),
        np.int32, "interpolated exponential",
    )
    exponentials = np.where(active, interpolated, 0).astype(np.int32)

    denominator = checked_integer(
        exponentials.astype(object).sum(axis=-1, keepdims=True),
        np.int64, "softmax denominator",
    )
    if np.any(denominator <= 0):
        raise ValueError("Nonpositive softmax denominator")

    # Rounded positive division, producing probability codes at 2^-16.
    numerator = checked_integer(
        exponentials.astype(object) * (1 << 16)
        + denominator.astype(object) // 2,
        np.int64, "probability numerator",
    )
    probabilities = checked_integer(
        numerator // denominator,
        np.int32, "probabilities",
    )

    assert np.all(
        (probabilities >= 0) & (probabilities <= (1 << 16))
    )
    assert np.all(np.where(allowed, 0, probabilities) == 0)

    row_sums = probabilities.astype(np.int64).sum(axis=-1)

    return {
        "shifted_scores": shifted,
        "coordinates": coordinates,
        "exponentials": exponentials,
        "denominator": denominator,
        "probabilities": probabilities,
        "max_row_sum_error": int(np.max(
            np.abs(row_sums - (1 << 16))
        )),
        "cutoff_count": int(np.count_nonzero(allowed & ~active)),
    }

def integer_attention_context(probabilities, value, params):
    if (
        probabilities.dtype != np.int32
        or probabilities.shape != (16, 32, 32)
    ):
        raise ValueError("Probabilities must be INT32 [16,32,32]")

    if value.dtype != np.int8 or value.shape != (1, 32, 128):
        raise ValueError("V must be INT8 [1,32,128]")

    if np.any(probabilities < 0) or np.any(probabilities > (1 << 16)):
        raise ValueError("Probability code outside [0,65536]")

    value_heads = (
        value.reshape(32, 16, 8)
        .transpose(1, 0, 2)
        .astype(np.int64)
    )
    value_heads -= params["value_zero_point"]

    # [16,32,32] @ [16,32,8] -> [16,32,8].
    context_heads = checked_integer(
        probabilities.astype(np.int64) @ value_heads,
        np.int32, "attention context",
    )

    # [head,token,dimension] -> [1,token,hidden].
    context_merged = (
        context_heads.transpose(1, 0, 2).reshape(1, 32, 128)
    )

    # Existing XREQUANT mode-2 conversion in the C implementation.
    converted = rq_checked(
        context_merged,
        params["context_multiplier"],
        params["context_shift"],
        "context requantization",
    )
    unbounded = checked_integer(
        converted.astype(object) + params["context_output_zero_point"],
        np.int64, "context output zero point",
    )

    return {
        "context_heads": context_heads,
        "context_merged": context_merged,
        "output": np.clip(unbounded, -128, 127).astype(np.int8),
        "saturation_count": int(np.count_nonzero(
            (unbounded < -128) | (unbounded > 127)
        )),
    }

def integer_out_residual(attention_output, input_codes, params):
    if input_codes.dtype != np.int8 or input_codes.shape != (1, 32, 128):
        raise ValueError("Residual input must be INT8 [1,32,128]")

    projection = integer_projection(attention_output, params)

    # Center the INT8 projection output, then convert into scale 2^-16.
    centered = (
        projection["output"].astype(object)
        - params["output_zero_point"]
    )
    branch = checked_integer(
        rq_checked(
            centered,
            params["branch_multiplier"],
            params["branch_shift"],
            "residual branch",
        ),
        np.int32, "stored residual branch",
    )

    # Exact alignment of the original block input.
    aligned_input = checked_integer(
        input_codes.astype(object) * params["input_alignment"],
        np.int32, "aligned block input",
    )

    # Both branches now use scale 2^-16.
    output = checked_integer(
        branch.astype(object) + aligned_input.astype(object),
        np.int32, "first residual sum",
    )

    return {
        "projection": projection,
        "branch": branch,
        "output": output,
    }

def requantize_to_i8(values, params, label):
    shifted = rq_checked(
        values, params["multiplier"], params["shift"], label
    )
    unbounded = checked_integer(
        shifted.astype(object) + params["output_zero_point"],
        np.int64, label + " zero point",
    )

    return {
        "output": np.clip(unbounded, -128, 127).astype(np.int8),
        "saturation_count": int(np.count_nonzero(
            (unbounded < -128) | (unbounded > 127)
        )),
    }

def integer_scalar_mul_i8(codes, params, label):
    centered = codes.astype(object) - params["input_zero_point"]
    product = checked_integer(
        centered * params["centered_constant"],
        np.int32, label + " product",
    )
    return requantize_to_i8(product, params, label)

def integer_gelu_front(input_codes, params):
    if input_codes.dtype != np.int8 or input_codes.shape != (1, 32, 512):
        raise ValueError("GELU expects INT8 [1,32,512]")

    half = integer_scalar_mul_i8(
        input_codes, params["half"], "GELU half"
    )

    centered = input_codes.astype(object) - params["input_zero_point"]
    cube_raw = checked_integer(
        centered ** 3, np.int32, "GELU cube"
    )
    cube = requantize_to_i8(
        cube_raw, params["cube"], "GELU cube conversion"
    )

    weighted = integer_scalar_mul_i8(
        cube["output"], params["weighted_cube"], "GELU weighted cube"
    )

    spec = params["sum"]
    linear_product = checked_integer(
        (input_codes.astype(object) - spec["linear_zero_point"])
        * spec["linear_multiplier"],
        np.int64, "GELU linear branch",
    )
    cubic_product = checked_integer(
        (weighted["output"].astype(object) - spec["cubic_zero_point"])
        * spec["cubic_multiplier"],
        np.int64, "GELU cubic branch",
    )
    combined = checked_integer(
        linear_product.astype(object) + cubic_product.astype(object),
        np.int64, "GELU combined branches",
    )

    # One rounded conversion after adding both scaled branches.
    added = requantize_to_i8(combined, spec, "GELU branch sum")

    argument = integer_scalar_mul_i8(
        added["output"], params["argument"], "GELU argument"
    )

    stages = {
        "half": half,
        "cube": cube,
        "weighted_cube": weighted,
        "sum": added,
        "argument": argument,
    }

    return {
        **{name: result["output"] for name, result in stages.items()},
        "cube_raw": cube_raw,
        "saturations": {
            name: result["saturation_count"]
            for name, result in stages.items()
        },
    }

def requantize_two_round_i8(values, params):
    if values.dtype != np.int32:
        raise ValueError("Two-round conversion expects INT32 input")

    multiplier = params["q31_multiplier"]
    shift = params["second_shift"]

    if not (
        0 < multiplier <= np.iinfo(np.int32).max
        and 0 <= shift <= 31
    ):
        raise ValueError("Invalid two-round parameters")

    high = rq_checked(
        values, multiplier, 31, "GELU Q31 output"
    )

    mask = (1 << shift) - 1
    remainder = high & mask
    threshold = (mask >> 1) + (high < 0).astype(np.int64)

    rounded = (
        (high >> shift)
        + (remainder > threshold).astype(np.int64)
    )

    unbounded = checked_integer(
        rounded.astype(object) + params["output_zero_point"],
        np.int64,
        "GELU final zero point",
    )

    return {
        "output": np.clip(unbounded, -128, 127).astype(np.int8),
        "saturation_count": int(np.count_nonzero(
            (unbounded < -128) | (unbounded > 127)
        )),
    }

def integer_gelu_back(front, params):
    indices = front["argument"].astype(np.int16) + 128
    tanh_codes = params["tanh_lut"][indices]

    # Tanh and one-plus-tanh both use scale 2^-24.
    factor_fine = checked_integer(
        tanh_codes.astype(object) + (1 << 24),
        np.int32,
        "one plus tanh",
    )

    factor = requantize_to_i8(
        factor_fine, params["factor"], "GELU factor"
    )

    final_params = params["final"]
    product = checked_integer(
        (front["half"].astype(object) - final_params["half_zero_point"])
        * (
            factor["output"].astype(object)
            - final_params["factor_zero_point"]
        ),
        np.int32,
        "GELU final product",
    )

    output = requantize_two_round_i8(product, final_params)

    return {
        "tanh": tanh_codes,
        "factor_fine": factor_fine,
        "factor": factor["output"],
        "product": product,
        "output": output["output"],
        "saturations": {
            "factor": factor["saturation_count"],
            "output": output["saturation_count"],
        },
    }

def integer_gelu(input_codes, params):
    front = integer_gelu_front(input_codes, params)
    back = integer_gelu_back(front, params)

    return {
        **front,
        **back,
        "saturations": {
            **front["saturations"],
            **back["saturations"],
        },
    }

def integer_mlp_out_residual(gelu_output, residual_input, params):
    if (
        residual_input.dtype != np.int32
        or residual_input.shape != (1, 32, 128)
    ):
        raise ValueError("Retained residual must be INT32 [1,32,128]")

    projection = integer_projection(gelu_output, params)

    centered = (
        projection["output"].astype(object)
        - params["output_zero_point"]
    )
    branch = checked_integer(
        rq_checked(
            centered,
            params["branch_multiplier"],
            params["branch_shift"],
            "final residual branch",
        ),
        np.int32, "stored final branch",
    )

    output_i32 = checked_integer(
        residual_input.astype(object) + branch.astype(object),
        np.int32, "final residual sum",
    )

    boundary = requantize_to_i8(
        output_i32, params["boundary"], "block output boundary"
    )

    return {
        "projection": projection,
        "branch": branch,
        "output_i32": output_i32,
        "output": boundary["output"],
        "boundary_saturations": boundary["saturation_count"],
    }


def integer_block_forward(input_codes, bundle):
    ln1 = integer_ln1(input_codes, bundle["ln1"])

    projections = {
        name: integer_projection(
            ln1["output"], bundle["qkv"][name]
        )
        for name in ("Q", "K", "V")
    }

    params = bundle["attention"]
    scores = integer_attention_scores(
        projections["Q"]["output"],
        projections["K"]["output"],
        params,
    )
    softmax = integer_causal_softmax(scores, params)
    attention = integer_attention_context(
        softmax["probabilities"],
        projections["V"]["output"],
        params,
    )

    residual = integer_out_residual(
        attention["output"],
        input_codes,
        bundle["out_residual"],
    )
    ln2 = integer_ln2(residual["output"], bundle["ln2"])

    expansion = integer_projection(
        ln2["output"], bundle["mlp_expand"]
    )
    gelu = integer_gelu(expansion["output"], bundle["gelu"])

    final = integer_mlp_out_residual(
        gelu["output"],
        residual["output"],
        bundle["mlp_out"],
    )

    return {
        "output": final["output"],
        "output_i32": final["output_i32"],
        "saturations": {
            "ln1": ln1["saturation_count"],
            **{
                name: projections[name]["saturation_count"]
                for name in ("Q", "K", "V")
            },
            "attention": attention["saturation_count"],
            "out_projection": residual["projection"]["saturation_count"],
            "ln2": ln2["saturation_count"],
            "mlp_expand": expansion["saturation_count"],
            **{
                "gelu_" + name: count
                for name, count in gelu["saturations"].items()
            },
            "mlp_out_projection": final["projection"]["saturation_count"],
            "block_boundary": final["boundary_saturations"],
        },
        "max_softmax_row_sum_error": softmax["max_row_sum_error"],
    }
