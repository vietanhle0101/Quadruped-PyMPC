"""Merge DPC dataset .npz files into a single dataset."""

import argparse
import pathlib
import shutil
import sys

import numpy as np


REQUIRED_KEYS = (
    "current_centroidal_state",
    "reference_state_horizon",
    "reference",
    "contact_sequence",
    "goal_base_pos",
    "initial_base_pos",
    "initial_base_rpy",
    "rollout_id",
    "time_index",
    "termination_code",
)


def collect_input_files(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    return []


def normalize_contact_sequence_shape(
    contact_sequence: np.ndarray,
    reference_state_horizon: np.ndarray,
) -> np.ndarray:
    if contact_sequence.ndim != 3:
        raise ValueError(
            f"contact_sequence should be 3D, got shape {contact_sequence.shape}"
        )

    expected_horizon = reference_state_horizon.shape[1]
    if contact_sequence.shape[1] == expected_horizon and contact_sequence.shape[2] == 4:
        return contact_sequence
    if contact_sequence.shape[1] == 4 and contact_sequence.shape[2] == expected_horizon:
        return np.transpose(contact_sequence, (0, 2, 1))

    raise ValueError(
        f"contact_sequence shape {contact_sequence.shape} is incompatible with "
        f"reference_state_horizon shape {reference_state_horizon.shape}"
    )


def inspect_dataset(path: pathlib.Path) -> dict[str, object]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")

        arrays = {key: data[key] for key in REQUIRED_KEYS}

    arrays["contact_sequence"] = normalize_contact_sequence_shape(
        arrays["contact_sequence"],
        arrays["reference_state_horizon"],
    )

    num_samples = int(arrays["current_centroidal_state"].shape[0])
    for key, array in arrays.items():
        if array.shape[0] != num_samples:
            raise ValueError(
                f"{path}: key {key} has {array.shape[0]} samples, expected {num_samples}"
            )

    return {
        "path": path,
        "num_samples": num_samples,
        "shapes": {key: arrays[key].shape[1:] for key in REQUIRED_KEYS},
        "dtypes": {key: arrays[key].dtype for key in REQUIRED_KEYS},
        "max_rollout_id": int(np.max(arrays["rollout_id"])) if num_samples > 0 else -1,
    }


def ensure_compatible_layout(metadata: list[dict[str, object]]) -> None:
    if not metadata:
        raise ValueError("No dataset files found to merge.")

    base_shapes = metadata[0]["shapes"]
    base_dtypes = metadata[0]["dtypes"]
    for item in metadata[1:]:
        for key in REQUIRED_KEYS:
            if item["shapes"][key] != base_shapes[key]:
                raise ValueError(
                    f"{item['path']}: shape mismatch for {key}: "
                    f"{item['shapes'][key]} vs {base_shapes[key]}"
                )
            if item["dtypes"][key] != base_dtypes[key]:
                raise ValueError(
                    f"{item['path']}: dtype mismatch for {key}: "
                    f"{item['dtypes'][key]} vs {base_dtypes[key]}"
                )


def merge_datasets(input_files: list[pathlib.Path], output_path: pathlib.Path) -> pathlib.Path:
    metadata = [inspect_dataset(path) for path in input_files]
    ensure_compatible_layout(metadata)

    total_samples = sum(item["num_samples"] for item in metadata)
    temp_dir = output_path.parent / f".{output_path.stem}_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    memmaps: dict[str, np.memmap] = {}
    try:
        for key in REQUIRED_KEYS:
            shape = (total_samples, *metadata[0]["shapes"][key])
            dtype = metadata[0]["dtypes"][key]
            memmaps[key] = np.lib.format.open_memmap(
                temp_dir / f"{key}.npy",
                mode="w+",
                dtype=dtype,
                shape=shape,
            )

        offset = 0
        rollout_offset = 0
        for item in metadata:
            path = item["path"]
            num_samples = item["num_samples"]
            print(f"Merging {path} ({num_samples} points)")

            with np.load(path) as data:
                reference_state_horizon = data["reference_state_horizon"]
                contact_sequence = normalize_contact_sequence_shape(
                    data["contact_sequence"],
                    reference_state_horizon,
                )

                for key in REQUIRED_KEYS:
                    if key == "contact_sequence":
                        source = contact_sequence
                    elif key == "rollout_id":
                        source = data[key] + rollout_offset
                    else:
                        source = data[key]
                    memmaps[key][offset : offset + num_samples] = source

            offset += num_samples
            rollout_offset += item["max_rollout_id"] + 1

        print(f"Writing merged dataset to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **{key: memmaps[key] for key in REQUIRED_KEYS},
        )
    finally:
        for mmap in memmaps.values():
            if hasattr(mmap, "flush"):
                mmap.flush()
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def main() -> int:
    default_input = pathlib.Path(__file__).resolve().parent.parent / "datasets"
    default_output = default_input / "dpc_dataset_merged.npz"

    parser = argparse.ArgumentParser(description="Merge DPC dataset .npz files into one file.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=pathlib.Path,
        default=default_input,
        help="A dataset .npz file or a directory containing dataset .npz files.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=default_output,
        help="Output path for the merged dataset.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print dataset lengths and total samples without merging.",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args.input_path)
    if not input_files:
        print(f"No dataset files found at: {args.input_path}")
        return 1

    if args.output_path in input_files:
        input_files = [path for path in input_files if path != args.output_path]

    if not input_files:
        print("No input files left to merge after excluding the output file.")
        return 1

    if args.list_only:
        metadata = [inspect_dataset(path) for path in input_files]
        total_samples = 0
        for item in metadata:
            print(f"{item['path']}: {item['num_samples']} points")
            total_samples += item["num_samples"]
        print(f"Total: {total_samples} points across {len(metadata)} file(s)")
        return 0

    merged_path = merge_datasets(input_files, args.output_path)
    print(f"Merged dataset saved to: {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
