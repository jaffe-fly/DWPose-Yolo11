import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def check_onnx_model(model_path: str):
    """
    Check ONNX model input/output information.

    Args:
        model_path: Path to ONNX model file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Checking model: {model_path.name}")
    print(f"{'=' * 60}\n")

    # Load model with ONNX
    try:
        onnx_model = onnx.load(str(model_path))
        print("✓ Model loaded successfully with ONNX")

        # Check model info
        print(f"\nModel IR Version: {onnx_model.ir_version}")
        print(f"Producer: {onnx_model.producer_name} {onnx_model.producer_version}")

        # Check inputs
        print(f"\n{'─' * 60}")
        print("INPUTS:")
        print(f"{'─' * 60}")
        for i, input_tensor in enumerate(onnx_model.graph.input):
            print(f"\nInput {i}: {input_tensor.name}")
            shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            print(f"  Shape: {shape}")
            print(f"  Type: {input_tensor.type.tensor_type.elem_type}")
            # Convert type number to name
            type_map = {
                1: "FLOAT32",
                2: "UINT8",
                3: "INT8",
                4: "UINT16",
                5: "INT16",
                6: "INT32",
                7: "INT64",
                8: "STRING",
                9: "BOOL",
                10: "FLOAT16",
                11: "DOUBLE",
                12: "UINT32",
                13: "UINT64",
            }
            type_name = type_map.get(input_tensor.type.tensor_type.elem_type, "UNKNOWN")
            print(f"  Type Name: {type_name}")

        # Check outputs
        print(f"\n{'─' * 60}")
        print("OUTPUTS:")
        print(f"{'─' * 60}")
        for i, output_tensor in enumerate(onnx_model.graph.output):
            print(f"\nOutput {i}: {output_tensor.name}")
            shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            print(f"  Shape: {shape}")
            print(f"  Type: {output_tensor.type.tensor_type.elem_type}")
            type_map = {
                1: "FLOAT32",
                2: "UINT8",
                3: "INT8",
                4: "UINT16",
                5: "INT16",
                6: "INT32",
                7: "INT64",
                8: "STRING",
                9: "BOOL",
                10: "FLOAT16",
                11: "DOUBLE",
                12: "UINT32",
                13: "UINT64",
            }
            type_name = type_map.get(
                output_tensor.type.tensor_type.elem_type, "UNKNOWN"
            )
            print(f"  Type Name: {type_name}")

    except Exception as e:
        print(f"Error loading model with ONNX: {e}")
        return

    # Also check with ONNX Runtime
    print(f"\n{'─' * 60}")
    print("ONNX Runtime Session Info:")
    print(f"{'─' * 60}")
    try:
        session = ort.InferenceSession(str(model_path))
        print("\nInputs:")
        for i, input_meta in enumerate(session.get_inputs()):
            print(f"  {i}: {input_meta.name}")
            print(f"     Shape: {input_meta.shape}")
            print(f"     Type: {input_meta.type}")

        print("\nOutputs:")
        for i, output_meta in enumerate(session.get_outputs()):
            print(f"  {i}: {output_meta.name}")
            print(f"     Shape: {output_meta.shape}")
            print(f"     Type: {output_meta.type}")
    except Exception as e:
        print(f"Error creating ONNX Runtime session: {e}")

    # Generate config.pbtxt snippet
    print(f"\n{'─' * 60}")
    print("Suggested config.pbtxt input/output section:")
    print(f"{'─' * 60}\n")

    print("input [")
    for input_tensor in onnx_model.graph.input:
        shape = [
            dim.dim_value if dim.dim_value > 0 else -1
            for dim in input_tensor.type.tensor_type.shape.dim
        ]
        # Remove batch dimension if it's 1 or dynamic
        if len(shape) > 0 and (shape[0] == 1 or shape[0] == -1):
            shape = shape[1:]

        type_map = {
            1: "TYPE_FP32",
            2: "TYPE_UINT8",
            3: "TYPE_INT8",
            10: "TYPE_FP16",
            11: "TYPE_FP64",
        }
        data_type = type_map.get(input_tensor.type.tensor_type.elem_type, "TYPE_FP32")

        print("  {")
        print(f'    name: "{input_tensor.name}"')
        print(f"    data_type: {data_type}")
        print(f"    dims: {shape}")
        print("  }")
    print("]")

    print("\noutput [")
    for output_tensor in onnx_model.graph.output:
        shape = [
            dim.dim_value if dim.dim_value > 0 else -1
            for dim in output_tensor.type.tensor_type.shape.dim
        ]
        # Remove batch dimension if it's 1 or dynamic
        if len(shape) > 0 and (shape[0] == 1 or shape[0] == -1):
            shape = shape[1:]

        type_map = {
            1: "TYPE_FP32",
            2: "TYPE_UINT8",
            3: "TYPE_INT8",
            10: "TYPE_FP16",
            11: "TYPE_FP64",
        }
        data_type = type_map.get(output_tensor.type.tensor_type.elem_type, "TYPE_FP32")

        print("  {")
        print(f'    name: "{output_tensor.name}"')
        print(f"    data_type: {data_type}")
        print(f"    dims: {shape}")
        print("  }")
    print("]")


def main():
    parser = argparse.ArgumentParser(
        description="Check ONNX model input/output format for Triton configuration"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to ONNX model file",
    )

    args = parser.parse_args()
    check_onnx_model(args.model)


if __name__ == "__main__":
    main()
