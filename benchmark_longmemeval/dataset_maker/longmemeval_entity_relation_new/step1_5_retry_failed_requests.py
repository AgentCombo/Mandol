from pathlib import Path
import runpy
import sys


def main() -> None:
    target = Path(__file__).with_name("step1.5_retry_failed_requests.py")
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
