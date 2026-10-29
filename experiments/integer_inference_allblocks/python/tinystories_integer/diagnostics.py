"""Host-only diagnostics: FP32 embeddings/head, LiteRT and greedy generation.

Imported only by full validation/generation commands, never by integer execution.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as torch_functional
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from .integer_reference import integer_block_forward
from .parameters import boundary_specs

def error_metrics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    difference = candidate - reference

    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(difference)
    relative_l2 = (
        100.0 * error_norm / reference_norm
        if reference_norm > 0
        else (0.0 if error_norm == 0 else float("inf"))
    )

    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2_percent": float(relative_l2),
    }

def encode_reference_block_output(values, boundary):
    # Diagnostic reference encoding using the runtime tie rule.
    rounded = np.floor(
        values.astype(np.float64) / boundary["scale"] + 0.5
    )
    return {
        "output": np.clip(rounded, -128, 127).astype(np.int8),
        "saturation_count": int(np.count_nonzero(
            (rounded < -128) | (rounded > 127)
        )),
    }

class HostReferences:
    def __init__(self, bundles, profile, models_dir, fp32_dir, load_tokenizer=False):
        self.bundles=bundles
        self.boundaries=boundary_specs(profile)
        self.int8_boundary_specs = [self.boundaries[0]['input']] + [b['output'] for b in self.boundaries]
        self.model=AutoModelForCausalLM.from_pretrained(
            fp32_dir, local_files_only=True, dtype=torch.float32).cpu().eval()
        self.model.set_attn_implementation("eager")
        if len(self.model.transformer.h)!=8:
            raise ValueError("Expected eight GPT-Neo blocks")
        self.tokenizer=AutoTokenizer.from_pretrained(fp32_dir,local_files_only=True) if load_tokenizer else None
        self.block_models={}
        for i in range(8):
            path=Path(models_dir)/f"block{i}_w8a8.tflite"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=profile["model_sha256"][str(i)]:
                raise ValueError(f"Block {i}: model hash mismatch")
            interpreter=Interpreter(model_path=str(path),experimental_preserve_all_tensors=True,
                experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
            interpreter.allocate_tensors()
            self.block_models[i]={"interpreter":interpreter,
                "input":interpreter.get_input_details()[0],"output":interpreter.get_output_details()[0],
                "tensors":{int(t["index"]):t for t in interpreter.get_tensor_details()},
                "ops":interpreter._get_ops_details()}

    def collect_fp32_boundaries(self, input_ids):
        ids = torch.as_tensor(np.asarray(input_ids).copy(), dtype=torch.long).reshape(1, 32)
        captured = {}
        handles = []

        def make_hook(block_index):

            def hook(module, args, output):
                hidden_output = output[0] if isinstance(output, (tuple, list)) else output
                captured[block_index] = {'input_fp32': args[0].detach().cpu().numpy().copy(), 'output_fp32': hidden_output.detach().cpu().numpy().copy()}
            return hook
        try:
            for block_index, block in enumerate(self.model.transformer.h):
                handles.append(block.register_forward_hook(make_hook(block_index)))
            with torch.no_grad():
                self.model.transformer(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 8:
            raise RuntimeError('Did not capture all eight blocks.')
        for block_index in range(8):
            for boundary in ('input', 'output'):
                values = captured[block_index][f'{boundary}_fp32']
                if values.shape != (1, 32, 128):
                    raise ValueError('Unexpected block boundary shape.')
                if not np.all(np.isfinite(values)):
                    raise ValueError('Non-finite reference activation.')
        return [captured[i] for i in range(8)]

    def run_tflite_block(self, block_index, values):
        model = self.block_models[block_index]
        interpreter = model['interpreter']
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected_shape = tuple((int(v) for v in model['input']['shape']))
        if values.shape != expected_shape:
            raise ValueError(f'Block {block_index}: unexpected input shape')
        interpreter.set_tensor(int(model['input']['index']), values)
        interpreter.invoke()
        output = interpreter.get_tensor(int(model['output']['index'])).copy()
        if not np.all(np.isfinite(output)):
            raise ValueError(f'Block {block_index}: non-finite output')
        return output

    def encode_boundary(self, values, edge):
        spec = self.int8_boundary_specs[edge]
        scale = spec['scale']
        zero_point = spec['zero_point']
        rounded = np.rint(values.astype(np.float64) / scale) + zero_point
        clipping_count = int(np.count_nonzero((rounded < -128) | (rounded > 127)))
        codes = np.clip(rounded, -128, 127).astype(np.int8)
        represented = ((codes.astype(np.float64) - zero_point) * scale).astype(np.float32)
        return (codes, represented, clipping_count)

    def predict_from_final_hidden(self, hidden_states):
        hidden = torch.from_numpy(np.ascontiguousarray(hidden_states, dtype=np.float32))
        with torch.no_grad():
            normalized = self.model.transformer.ln_f(hidden)
            logits = self.model.get_output_embeddings()(normalized)
            predictions = logits.argmax(dim=-1)
        return predictions.cpu().numpy().reshape(-1)

    def logits_from_final_hidden(self, hidden_states):
        hidden = torch.from_numpy(np.ascontiguousarray(hidden_states, dtype=np.float32))
        with torch.no_grad():
            normalized = self.model.transformer.ln_f(hidden)
            return self.model.get_output_embeddings()(normalized)

    def next_token_losses(self, hidden_states, input_ids):
        ids = np.asarray(input_ids, dtype=np.int64)
        assert ids.shape == (1, 32)
        logits = self.logits_from_final_hidden(hidden_states)
        prediction_logits = logits[:, :-1, :].double()
        targets = torch.from_numpy(ids[:, 1:].copy()).reshape(-1)
        losses = torch_functional.cross_entropy(prediction_logits.reshape(-1, prediction_logits.shape[-1]), targets, reduction='none').cpu().numpy().copy()
        assert losses.size == 31
        assert np.all(np.isfinite(losses))
        return losses

    def prepare_generation_input(self, token_ids):
        length = len(token_ids)
        if not 1 <= length <= 32:
            raise ValueError('The generation prefix must contain 1..32 tokens')
        padded_ids = np.full((1, 32), self.tokenizer.eos_token_id, dtype=np.int64)
        padded_ids[0, :length] = token_ids
        ids = torch.from_numpy(padded_ids)
        positions = torch.arange(32, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.model.transformer.drop(self.model.transformer.wte(ids) + self.model.transformer.wpe(positions)).cpu().numpy().copy()
        return {'input_ids': padded_ids, 'embeddings': embeddings, 'length': length, 'prediction_position': length - 1}

    def generation_next_logits(self, path, frame):
        position = frame['prediction_position']
        diagnostics = {}
        if path == 'fp32':
            with torch.no_grad():
                logits = self.model(input_ids=torch.from_numpy(frame['input_ids']), position_ids=torch.arange(32, dtype=torch.long).unsqueeze(0), use_cache=False, return_dict=True).logits
            return (logits[:, position, :], diagnostics)
        if path == 'direct':
            hidden = frame['embeddings'].copy()
            for block_index in range(8):
                hidden = self.run_tflite_block(block_index, hidden)
        elif path == 'mixed_boundary':
            _, hidden, input_clips = self.encode_boundary(frame['embeddings'], 0)
            boundary_clips = []
            for block_index in range(8):
                raw = self.run_tflite_block(block_index, hidden)
                boundary = self.bundles[block_index]['mlp_out']['boundary']
                encoded = encode_reference_block_output(raw, boundary)
                boundary_clips.append(encoded['saturation_count'])
                hidden = (encoded['output'].astype(np.float64) * boundary['scale']).astype(np.float32)
            diagnostics = {'initial_input_clips': input_clips, 'boundary_clips': boundary_clips}
        elif path == 'integer':
            codes, _, input_clips = self.encode_boundary(frame['embeddings'], 0)
            boundary_clips = []
            for block_index in range(8):
                result = integer_block_forward(codes, self.bundles[block_index])
                codes = result['output']
                boundary_clips.append(result['saturations']['block_boundary'])
            diagnostics = {'initial_input_clips': input_clips, 'boundary_clips': boundary_clips, 'final_codes': codes.copy()}
            hidden = (codes.astype(np.float64) * self.boundaries[7]['output']['scale']).astype(np.float32)
        else:
            raise ValueError(f'Unknown generation path: {path}')
        logits = self.logits_from_final_hidden(hidden)
        return (logits[:, position, :], diagnostics)

    def generate_with_path(self, path, prompt_ids, max_new_tokens=20):
        token_ids = list(prompt_ids)
        if not 1 <= len(token_ids) < 32:
            raise ValueError('Choose a prompt containing 1..31 tokens')
        budget = min(max_new_tokens, 32 - len(token_ids))
        steps = []
        stop_reason = 'context_limit' if budget < max_new_tokens else 'token_limit'
        for step_index in range(budget):
            frame = self.prepare_generation_input(token_ids)
            logits, diagnostics = self.generation_next_logits(path, frame)
            next_id = int(logits.argmax(dim=-1).item())
            top_two = torch.topk(logits, k=2, dim=-1)
            margin = float((top_two.values[0, 0] - top_two.values[0, 1]).item())
            token_ids.append(next_id)
            piece = self.tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
            steps.append({'token_id': next_id, 'piece': piece, 'top_two_logit_margin': margin, 'diagnostics': diagnostics})
            print(f'{path}, step {step_index + 1:02d}: {piece!r}', flush=True)
            if next_id == self.tokenizer.eos_token_id:
                stop_reason = 'eos'
                break
        return {'token_ids': token_ids, 'generated_token_ids': token_ids[len(prompt_ids):], 'steps': steps, 'stop_reason': stop_reason, 'text': self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)}

    def generate_with_32_token_window(self, path, prompt_ids, max_new_tokens=20):
        if len(prompt_ids) != 32:
            raise ValueError('This test requires exactly 32 prompt tokens')
        token_ids = list(prompt_ids)
        steps = []
        stop_reason = 'token_limit'
        for step_index in range(max_new_tokens):
            context_ids = token_ids[-32:]
            frame = self.prepare_generation_input(context_ids)
            assert frame['length'] == 32
            logits, diagnostics = self.generation_next_logits(path, frame)
            next_id = int(logits.argmax(dim=-1).item())
            top_two = torch.topk(logits, k=2, dim=-1)
            margin = float((top_two.values[0, 0] - top_two.values[0, 1]).item())
            piece = self.tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
            steps.append({'context_ids': context_ids.copy(), 'token_id': next_id, 'piece': piece, 'top_two_logit_margin': margin, 'diagnostics': diagnostics})
            token_ids.append(next_id)
            print(f'{path}, step {step_index + 1:02d}, context=32: {piece!r}', flush=True)
            if next_id == self.tokenizer.eos_token_id:
                stop_reason = 'eos'
                break
        return {'token_ids': token_ids, 'generated_token_ids': token_ids[32:], 'steps': steps, 'stop_reason': stop_reason, 'text': self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)}

