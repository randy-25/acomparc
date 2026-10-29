#!/usr/bin/env python3
"""Eight-block export extracted from the project notebook."""

def main():
    from pathlib import Path
    
    
    import argparse
    
    PROJECT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export eight TinyStories W8A8 TFLite blocks."
    )
    parser.add_argument(
        "--fp32-dir", type=Path,
        default=PROJECT / "models/tinystories_3m_fp32",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        help="New output directory; defaults to a timestamped model directory.",
    )
    args = parser.parse_args()
    
    
    # Standard library
    import copy
    import hashlib
    import json
    import sys
    import importlib.metadata as package_metadata
    
    from pathlib import Path
    from collections import Counter
    from datetime import datetime, timezone
    
    # Numerical computation and original model
    import numpy as np
    import torch
    
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # TFLite / LiteRT execution
    import litert_torch
    
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    
    # Quantization and model export
    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e,
    )
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )
    from litert_torch.quantize.quant_config import QuantConfig
    
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Imports loaded.")
    
    
    MODEL_ID = "roneneldan/TinyStories-3M"
    MODEL_REVISION = "cfaf26ec85ecdfc1bd7c2638104cce55cb67f894"
    
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        args.fp32_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.fp32_dir,
        local_files_only=True,
    )
    
    
    # Input specification
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 32
    GENERATED_LENGTH = 20
    
    PROMPT = (
        "Once upon a time, a curious little girl named Lily lived in a quiet "
        "village beside a green forest. Every morning, she carried fresh bread "
        "to her grandmother and watched the birds in the trees."
    )
    
    # Verify the loaded model's architecture.
    config = model_fp32.config
    
    assert config.num_layers == 8
    assert config.hidden_size == 128
    assert config.num_heads == 16
    
    HEAD_DIM = config.hidden_size // config.num_heads
    MLP_HIDDEN_SIZE = config.intermediate_size or 4 * config.hidden_size
    
    assert HEAD_DIM == 8
    assert MLP_HIDDEN_SIZE == 512
    
    BLOCK_INPUT_SHAPE = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        config.hidden_size,
    )
    
    # Choices to apply when we construct the quantizer.
    QUANTIZATION_SETTINGS = {
        "scheme": "W8A8",
        "is_per_channel": False,
        "is_dynamic": False,
        "is_qat": False,
    }
    
    # Used later for full-model generation tests.
    GENERATION_SETTINGS = {
        "max_new_tokens": GENERATED_LENGTH,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    
    print("Block input shape:", BLOCK_INPUT_SHAPE)
    print("Head dimension:", HEAD_DIM)
    print("MLP hidden size:", MLP_HIDDEN_SIZE)
    print("Quantization settings:", QUANTIZATION_SETTINGS)
    
    # Working copy for export; the original model stays available.
    export_model_fp32 = copy.deepcopy(model_fp32).cpu().eval()
    export_model_fp32.set_attn_implementation("eager")
    
    assert len(export_model_fp32.transformer.h) == 8
    assert BATCH_SIZE == 1
    assert SEQUENCE_LENGTH == 32
    
    def collect_block_io(input_ids):
        """Capture the actual FP32 input and output of every block."""
        captured = {}
        handles = []
    
        def make_hook(block_index):
            def hook(module, args, output):
                hidden_output = (
                    output[0] if isinstance(output, (tuple, list)) else output
                )
                captured[block_index] = (
                    args[0].detach().clone(),
                    hidden_output.detach().clone(),
                )
            return hook
    
        try:
            for i, block in enumerate(export_model_fp32.transformer.h):
                handles.append(block.register_forward_hook(make_hook(i)))
    
            with torch.no_grad():
                export_model_fp32.transformer(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
    
        assert len(captured) == 8
        return [captured[i] for i in range(8)]
    
    # Fixed benchmark prompt from our specification cell.
    benchmark_ids = tokenizer(
        PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    
    assert benchmark_ids.shape[1] >= SEQUENCE_LENGTH
    benchmark_ids = benchmark_ids[:, :SEQUENCE_LENGTH]
    benchmark_block_io = collect_block_io(benchmark_ids)
    
    # Small initial calibration corpus.
    calibration_prompts = [
        "Once upon a time, a little girl named Lily lived beside a quiet "
        "forest. Every morning she carried bread to her grandmother and "
        "greeted all the animals along the path.",
    
        "Tom found a small red ball underneath his bed. He took it outside "
        "and played with his dog until the sun disappeared behind the hills.",
    
        "A young rabbit wanted to cross the river, but the water was too "
        "deep. His friends gathered branches and helped him build a bridge.",
    
        "Mia planted a tiny seed in the garden. She watered it every day, "
        "and soon a tall sunflower grew beside her bedroom window.",
    
        "The little bird could not fly as high as the others. She practiced "
        "every afternoon until she finally reached the top of the old tree.",
    
        "Ben heard a strange sound inside the kitchen. He opened the door "
        "slowly and discovered a hungry kitten looking for some food.",
    
        "A kind farmer shared his vegetables with the people in his village. "
        "When winter arrived, everyone helped him repair his wooden barn.",
    
        "Sara lost her favorite blue hat at the park. She searched near the "
        "swings and found it beside a sleeping brown puppy.",
    ]
    
    # Concatenate stories with EOS separators, then take full 32-token chunks.
    assert tokenizer.eos_token_id is not None
    calibration_tokens = []
    
    for text in calibration_prompts:
        calibration_tokens.extend(
            tokenizer.encode(text, add_special_tokens=False)
        )
        calibration_tokens.append(tokenizer.eos_token_id)
    
    calibration_ids = [
        torch.tensor(
            [calibration_tokens[start:start + SEQUENCE_LENGTH]],
            dtype=torch.long,
        )
        for start in range(
            0,
            len(calibration_tokens) - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH,
        )
    ]
    
    assert calibration_ids
    calibration_block_inputs = [[] for _ in range(8)]
    
    for ids in calibration_ids:
        block_io = collect_block_io(ids)
        for i, (block_input, _) in enumerate(block_io):
            calibration_block_inputs[i].append(block_input)
    
    print("Calibration sequences:", len(calibration_ids))
    
    for i, (block_input, _) in enumerate(benchmark_block_io):
        assert tuple(block_input.shape) == (1, 32, 128)
        print(
            f"Block {i}: input {tuple(block_input.shape)}, "
            f"{len(calibration_block_inputs[i])} calibration samples"
        )
    
    class BlockExportWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
    
        def forward(self, hidden_states):
            output = self.block(
                hidden_states,
                layer_past=None,
                attention_mask=None,
                use_cache=False,
                output_attentions=False,
            )
            return output[0] if isinstance(output, (tuple, list)) else output
    
    
    def as_numpy(output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        return np.asarray(output)
    
    
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    
    QUANTIZED_BLOCK_DIR = args.out_dir or (PROJECT / 'models/tinystories_3m_blocks_w8a8' / run_name)
    QUANTIZED_BLOCK_DIR.mkdir(parents=True, exist_ok=False)
    
    report = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "sequence_length": SEQUENCE_LENGTH,
        "use_cache": False,
        "attention_implementation": "eager",
        "quantization": {
            "is_per_channel": False,
            "is_dynamic": False,
            "is_qat": False,
            "fold_quantize": False,
            "fully_integer": False,
        },
        "calibration_prompts": calibration_prompts,
        "calibration_sequences": len(calibration_ids),
        "calibration_source": "preceding FP32 blocks",
        "benchmark_prompt": PROMPT,
        "versions": {
            name: package_metadata.version(name)
            for name in [
                "torch", "transformers", "torchao",
                "litert-torch", "ai-edge-litert", "numpy",
            ]
        },
        "blocks": [],
    }
    
    # Preserve the token sequences and per-block benchmark references.
    validation_data = {
        "benchmark_input_ids": benchmark_ids.numpy(),
        "calibration_input_ids": torch.cat(calibration_ids).numpy(),
    }
    for i, (x, y) in enumerate(benchmark_block_io):
        validation_data[f"block{i}_input_fp32"] = x.numpy()
        validation_data[f"block{i}_output_fp32"] = y.numpy()
    
    np.savez(
        QUANTIZED_BLOCK_DIR / "validation_inputs.npz",
        **validation_data,
    )
    
    print("Saving to:", QUANTIZED_BLOCK_DIR, flush=True)
    
    for i in range(8):
        print(f"\nQuantizing block {i}...", flush=True)
    
        wrapper = BlockExportWrapper(
            copy.deepcopy(export_model_fp32.transformer.h[i])
        ).eval()
    
        example_input, native_reference = benchmark_block_io[i]
        sample_args = (example_input,)
    
        # Check that isolated execution matches the block in the full model.
        with torch.no_grad():
            isolated_reference = wrapper(example_input)
    
        torch.testing.assert_close(
            isolated_reference,
            native_reference,
            atol=1e-5,
            rtol=1e-5,
        )
    
        exported = torch.export.export(wrapper, sample_args)
        with torch.no_grad():
            torch.testing.assert_close(
                exported.module()(example_input),
                isolated_reference,
            )
    
        quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=False,
                is_dynamic=False,
                is_qat=False,
            )
        )
    
        prepared = prepare_pt2e(exported.module(), quantizer)
    
        with torch.no_grad():
            for calibration_input in calibration_block_inputs[i]:
                prepared(calibration_input)
    
        quantized = convert_pt2e(prepared, fold_quantize=False)
    
        edge_block = litert_torch.convert(
            quantized,
            sample_args,
            quant_config=QuantConfig(pt2e_quantizer=quantizer),
        )
    
        output_before_save = as_numpy(edge_block(example_input))
        model_path = QUANTIZED_BLOCK_DIR / f"block{i}_w8a8.tflite"
        edge_block.export(str(model_path))
    
        # Verify that the saved file reloads with the same output.
        reloaded = litert_torch.load(str(model_path))
        output_after_save = as_numpy(reloaded(example_input))
    
        np.testing.assert_allclose(
            output_after_save,
            output_before_save,
            atol=1e-6,
            rtol=1e-6,
        )
    
        error = np.abs(
            output_after_save - native_reference.numpy()
        )
        assert np.isfinite(output_after_save).all()
    
        # Inspect the exported tensor types and projection quantization.
        interpreter = Interpreter(
            model_path=str(model_path),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        tensors = {
            t["index"]: t for t in interpreter.get_tensor_details()
        }
        dtype_counts = dict(Counter(
            np.dtype(t["dtype"]).name for t in tensors.values()
        ))
    
        int8_fc_count = 0
        for op in interpreter._get_ops_details():
            if op["op_name"] == "FULLY_CONNECTED":
                indices = [
                    int(op["inputs"][0]),
                    int(op["inputs"][1]),
                    int(op["outputs"][0]),
                ]
                if all(
                    tensors[index]["dtype"] == np.int8
                    for index in indices
                ):
                    int8_fc_count += 1
    
        assert int8_fc_count == 6, (
            f"Block {i}: expected six INT8 projections; "
            f"found {int8_fc_count}. Inspect this export."
        )
    
        block_report = {
            "block_index": i,
            "file": model_path.name,
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "bytes": model_path.stat().st_size,
            "int8_fully_connected_ops": int8_fc_count,
            "tensor_dtype_counts": dtype_counts,
            "benchmark_max_abs_error": float(error.max()),
            "benchmark_mean_abs_error": float(error.mean()),
            "reload_check": "passed",
        }
        report["blocks"].append(block_report)
    
        (QUANTIZED_BLOCK_DIR / "export_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    
        print(
            f"Saved {model_path.name} | INT8 FCs: {int8_fc_count} | "
            f"max error: {error.max():.6f} | "
            f"mean error: {error.mean():.6f}",
            flush=True,
        )
        print("Tensor types:", dtype_counts, flush=True)
    
    print("\nSaved all eight blocks to:")
    print(QUANTIZED_BLOCK_DIR)


if __name__ == "__main__":
    main()
