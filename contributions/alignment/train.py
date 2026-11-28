"""Train only the fixed width32/160 A/B comparison, with explicit input paths."""
import argparse
import json
from pathlib import Path
import torch

from . import adapter as a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    a.add_paths(parser)
    parser.add_argument("--seeds", type=int, nargs="+", choices=a.SEEDS, required=True)
    parser.add_argument("--roles", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--stop-after", type=int, choices=(1, 3, 8), help="GPU preflight only; config remains 160 epochs")
    parser.add_argument("--preflight-proof", help="passed GPU continuous/resumed A/B3 proof required for full runs")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.roles)) != len(args.roles):
        parser.error("Duplicate seeds/roles are not allowed")
    paths = a.Paths.from_args(args)
    identity = a.prepare_output(paths)
    if args.stop_after is None:
        if not args.preflight_proof:
            parser.error("Complete the A/B three-epoch GPU resume check and provide --preflight-proof before full training")
        proof = a.read(args.preflight_proof)
        if proof.get("kind") != "gpu_continuous_resume_AB3" or not proof.get("passed") or proof.get("identity") != identity or proof.get("checker_sha256") != a.sha(a.PACKAGE / "check.py"):
            raise ValueError("GPU preflight proof does not match these frozen inputs/sources")
    with a.process_lock(paths.output):
        for seed in args.seeds:
            for role in args.roles:
                cfg = a.config(seed, role)
                name = f"{role}_seed{seed}"
                folder = paths.output / "runs" / name
                if args.stop_after and (folder / "last.pt").exists():
                    previous = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                    if previous["epoch"] >= args.stop_after:
                        raise ValueError("This preflight already reached its stop boundary; use another output or a later boundary")
                    del previous
                a.write(paths.output / "progress.json", {"state": "training", "seed": seed, "role": role, "run": name, "preflight_only": args.stop_after is not None})
                with a.training_scope(paths):
                    result = a.core.run(cfg, name, stop_after=args.stop_after)
                if a.input_identity(paths) != identity:
                    raise ValueError("Frozen source/data/Teacher initialization inputs changed during this call")
                rows = a.history(folder, require_complete=args.stop_after is None)
                a.write(folder / "portable_provenance.json", {"identity": identity, "legacy_source_identity_is_not_claimed": True, "epochs_observed": len(rows), "run": name, "config": cfg})
                print(json.dumps({"run": name, "result": result, "observed_epochs": len(rows)}, ensure_ascii=False), flush=True)
        a.write(paths.output / "progress.json", {"state": "preflight_training_stopped" if args.stop_after else "training_complete_validation_pending", "seeds": args.seeds, "roles": args.roles, "manual_review_complete": False})


if __name__ == "__main__":
    main()
