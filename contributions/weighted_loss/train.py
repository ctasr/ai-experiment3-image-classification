"""Train/resume one explicitly named mode; never schedules another seed or test."""
from pathlib import Path
import argparse
import json

from . import adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=adapter.MODES, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 17, 73), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-cutmix", type=Path, required=True)
    parser.add_argument("--teacher-transfer", type=Path, required=True)
    parser.add_argument("--imagenet-weights", type=Path, help="Defaults to DATA_DIR/pretrained/resnet18-f37072fd.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after", type=int, default=240)
    parser.add_argument("--pilot-samples", type=int, choices=(456,))
    args = parser.parse_args(argv)
    config = args.config or Path(__file__).with_name("configs") / (args.mode + ".json")
    result = adapter.run(config, args.mode, args.seed, args.data_dir, args.teacher_cutmix,
                         args.teacher_transfer, args.output_dir, args.run_id, args.stop_after, args.pilot_samples, args.imagenet_weights)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
