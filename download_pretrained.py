"""下载PyTorch官方ResNet18权重，完整SHA256校验后供离线训练使用。"""
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PATH = Path(__file__).resolve().parent / "data/pretrained/resnet18-f37072fd.pth"


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if PATH.exists() and hashlib.sha256(PATH.read_bytes()).hexdigest() == SHA256:
        print("预训练权重已存在且校验通过。")
        return
    temporary = PATH.with_suffix(".download")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL, timeout=30) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == SHA256, "权重校验失败"
            temporary.replace(PATH)
            print("官方预训练权重下载并校验完成。")
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


if __name__ == "__main__":
    main()
