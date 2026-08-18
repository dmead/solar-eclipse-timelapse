"""Job log writer — the contract every log tailer in the old pipeline
expects (pjsr/ser-stack.js:44-63): `[ISO timestamp] message` lines, flushed
per line, with machine-readable `PROGRESS <k>/<n> <stage>` lines and
`=== <KIND> OK` / `*** <KIND> FAILED:` sentinels."""

import datetime
import os
import sys


class JobLog:
    def __init__(self, path: str, echo: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", encoding="utf-8", newline="\n")
        self._echo = echo

    def log(self, msg: str) -> None:
        stamp = datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.") + f"{datetime.datetime.now(datetime.UTC).microsecond // 1000:03d}Z"
        line = f"[{stamp}] {msg}"
        self._f.write(line + "\n")
        self._f.flush()
        if self._echo:
            print(line, file=sys.stderr)

    def progress(self, k: int, n: int, stage: str) -> None:
        self.log(f"PROGRESS {k}/{n} {stage}")

    def close(self) -> None:
        self._f.close()
