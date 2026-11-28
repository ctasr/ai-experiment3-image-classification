"""Run only one explicitly requested, gated stage. No command auto-fetches."""
import argparse
from pathlib import Path
import traceback
from . import common as C


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare-manifest","preflight","seal","fetch","evaluate","analyze"))
    p.add_argument("--output-dir",required=True,type=Path,help="Independent output directory, never a historical run directory")
    p.add_argument("--bindings",type=Path,help="Explicit 21-checkpoint/source JSON; required for prepare-manifest")
    p.add_argument("--data-dir",type=Path,help="Original project data directory containing raw/, splits/, dataset_manifest.json and normalization.json")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="prepare-manifest" and (args.bindings is None or args.data_dir is None):
        parser().error("prepare-manifest requires --bindings and --data-dir")
    if args.stage!="prepare-manifest" and (args.bindings is not None or args.data_dir is not None):
        parser().error("Bindings/data paths are frozen by prepare-manifest, not overridden in later stages")
    with C.output_lock(args.output_dir):
        try:
            if args.stage=="prepare-manifest":
                from .manifest import prepare
                result=prepare(args.bindings,args.data_dir,args.output_dir)
                print(f"Prepared {result['ready_checkpoints']}/21 fixed checkpoint bindings; complete={result['complete']}; no external access")
            elif args.stage=="preflight":
                from .inference import preflight
                preflight(args.output_dir);print("Actual original-validation GPU evaluator preflight passed; no external access")
            elif args.stage=="seal":
                from .manifest import seal
                seal(args.output_dir);print("Fixed 21-checkpoint / 24-row confirmation sealed; dataset has not been fetched")
            elif args.stage=="fetch":
                from .dataset import fetch
                fetch(args.output_dir);print("Pinned single external dataset fetched and byte/schema identity checked")
            elif args.stage=="evaluate":
                from .inference import evaluate
                evaluate(args.output_dir);print("Fixed FP32 logit evaluation complete; analysis/manual figures pending")
            elif args.stage=="analyze":
                from .analysis import analyze
                analyze(args.output_dir);print("Fixed analysis complete; external dataset retired; actual manual figure review remains pending")
        except Exception:
            from .manifest import utc
            C.write(args.output_dir/(args.stage+"_failure_"+utc().replace(":","-")+".json"),
                    {"stage":args.stage,"traceback":traceback.format_exc(),"existing_artifacts_preserved":True})
            raise


if __name__=="__main__":
    main()
