"""Augment a DPC dataset by adding Gaussian noise to current_centroidal_state."""

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


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


def load_dataset(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        return {key: np.asarray(data[key]) for key in REQUIRED_KEYS}


def build_noise_std(
    position_std: float,
    linear_velocity_std: float,
    orientation_std: float,
    angular_velocity_std: float,
    foot_position_std: float,
) -> np.ndarray:
    std = np.zeros((24,), dtype=np.float32)
    std[0:3] = position_std
    std[3:6] = linear_velocity_std
    std[6:9] = orientation_std
    std[9:12] = angular_velocity_std
    std[12:24] = foot_position_std
    return std


def augment_current_state(
    current_state: np.ndarray,
    rng: np.random.Generator,
    noise_std: np.ndarray,
) -> np.ndarray:
    if current_state.ndim != 2 or current_state.shape[1] != 24:
        raise ValueError(
            "current_centroidal_state must have shape (N, 24), "
            f"got {current_state.shape}"
        )
    noise = rng.normal(loc=0.0, scale=noise_std, size=current_state.shape).astype(np.float32)
    return current_state.astype(np.float32) + noise


def concatenate_datasets(
    original: dict[str, np.ndarray],
    noisy_current_state: np.ndarray,
) -> dict[str, np.ndarray]:
    augmented = {}
    for key, array in original.items():
        if key == "current_centroidal_state":
            augmented[key] = np.concatenate(
                (array.astype(np.float32), noisy_current_state),
                axis=0,
            )
        else:
            augmented[key] = np.concatenate((array, array), axis=0)
    return augmented


def replace_current_state_only(
    original: dict[str, np.ndarray],
    noisy_current_state: np.ndarray,
) -> dict[str, np.ndarray]:
    augmented = dict(original)
    augmented["current_centroidal_state"] = noisy_current_state
    return augmented


def save_dataset(dataset: dict[str, np.ndarray], output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return output_path


def main() -> int:
    default_input = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "dpc_dataset_1M.npz"

    parser = argparse.ArgumentParser(
        description="Augment a DPC dataset by adding Gaussian noise to current_centroidal_state."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=pathlib.Path,
        default=default_input,
        help="Input dataset .npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=None,
        help="Output path for the augmented dataset. Defaults to '<input>_noise.npz'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise generation.")
    parser.add_argument("--position-std", type=float, default=0.05, help="Std for base position noise.")
    parser.add_argument("--orientation-std", type=float, default=0.2, help="Std for base orientation noise.")
    parser.add_argument("--linear-velocity-std", type=float, default=0.2, help="Std for base linear velocity noise.")
    parser.add_argument("--angular-velocity-std", type=float, default=0.2, help="Std for base angular velocity noise.")
    parser.add_argument("--foot-position-std", type=float, default=0.05, help="Std for foot position noise.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append a noisy copy to the original dataset instead of replacing current states in place.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        print(f"Input dataset not found: {input_path}")
        return 1

    output_path = args.output_path
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_noise.npz")
    elif not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    dataset = load_dataset(input_path)
    rng = np.random.default_rng(args.seed)
    noise_std = build_noise_std(
        position_std=args.position_std,
        linear_velocity_std=args.linear_velocity_std,
        orientation_std=args.orientation_std,
        angular_velocity_std=args.angular_velocity_std,
        foot_position_std=args.foot_position_std,
    )
    noisy_current_state = augment_current_state(dataset["current_centroidal_state"], rng, noise_std)

    if args.append:
        augmented = concatenate_datasets(dataset, noisy_current_state)
    else:
        augmented = replace_current_state_only(dataset, noisy_current_state)

    save_dataset(augmented, output_path)

    original_points = int(dataset["current_centroidal_state"].shape[0])
    augmented_points = int(augmented["current_centroidal_state"].shape[0])
    mode = "append" if args.append else "replace"
    print(f"Input dataset: {input_path}")
    print(f"Output dataset: {output_path}")
    print(f"Augmentation mode: {mode}")
    print(f"Original points: {original_points}")
    print(f"Output points: {augmented_points}")
    print(f"Noise std: {noise_std.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
