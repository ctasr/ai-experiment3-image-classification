"""第三轮训练入口，逐轮训练完全复用已验证的第二轮引擎。"""
import argparse
import json
from data import ROOT
import extensions.train as engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    engine.DIRECTORY = ROOT / "runs/round3"
    engine.run(json.loads(open(args.config, encoding="utf-8").read()), args.run_id)


if __name__ == "__main__":
    main()
