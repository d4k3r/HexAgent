"""Framed GTP client and parser for completed KataHex analysis responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import selectors
import subprocess
import time
from typing import BinaryIO, Iterable


class GTPError(RuntimeError):
    pass


class GTPTimeout(GTPError):
    pass


class GTPProtocolError(GTPError):
    def __init__(self, response: "GTPResponse") -> None:
        super().__init__(f"GTP command {response.command_id} failed: {response.payload}")
        self.response = response


class MoveKind(str, Enum):
    PHYSICAL = "physical"
    PASS = "pass"
    RESIGN = "resign"
    SWAP = "swap"


def classify_move(move: str) -> MoveKind:
    normalized = move.strip().lower()
    if normalized == "pass":
        return MoveKind.PASS
    if normalized == "resign":
        return MoveKind.RESIGN
    if normalized == "swap":
        return MoveKind.SWAP
    return MoveKind.PHYSICAL


@dataclass(frozen=True, slots=True)
class GTPResponse:
    command_id: int | None
    success: bool
    payload: str
    raw: str


@dataclass(frozen=True, slots=True)
class AnalysisCandidate:
    move: str
    visits: int | None
    winrate: float | None
    utility: float | None
    prior: float | None
    lcb: float | None
    utility_lcb: float | None
    pv: tuple[str, ...]
    weight: float | None = None
    order: int | None = None

    @property
    def move_kind(self) -> MoveKind:
        return classify_move(self.move)


@dataclass(frozen=True, slots=True)
class CompletedAnalysis:
    candidates: tuple[AnalysisCandidate, ...]
    chosen_move: str
    raw_response: str

    @property
    def chosen_move_kind(self) -> MoveKind:
        return classify_move(self.chosen_move)


_FRAME_START = re.compile(br"(?m)(?:^|\n)([=?])(\d*) ?")
_CANDIDATE = re.compile(r"(?:^|\s)info move (\S+)\s+(.*?)(?=(?:\s+info move\s)|(?:\nplay\s)|\Z)", re.DOTALL)
_SCALAR_FIELDS = {
    "visits": int,
    "winrate": float,
    "utility": float,
    "prior": float,
    "lcb": float,
    "utilityLcb": float,
    "weight": float,
    "order": int,
}
_PV_END_FIELDS = (
    "pvVisits",
    "pvEdgeVisits",
    "ownership",
    "ownershipStdev",
    "movesOwnership",
)


def _scalar(text: str, name: str, conversion):
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", text)
    if match is None:
        return None
    try:
        return conversion(match.group(1))
    except ValueError as error:
        raise GTPError(f"invalid {name} in analysis: {match.group(1)}") from error


def parse_completed_analysis(response: GTPResponse) -> CompletedAnalysis:
    if not response.success:
        raise GTPProtocolError(response)
    candidates: list[AnalysisCandidate] = []
    for match in _CANDIDATE.finditer(response.payload):
        move, fields = match.groups()
        pv_match = re.search(
            rf"(?:^|\s)pv\s+(.+?)(?=\s+(?:{'|'.join(_PV_END_FIELDS)})\s|\Z)",
            fields,
            re.DOTALL,
        )
        pv = tuple(pv_match.group(1).split()) if pv_match else ()
        candidates.append(
            AnalysisCandidate(
                move=move.lower(),
                visits=_scalar(fields, "visits", int),
                winrate=_scalar(fields, "winrate", float),
                utility=_scalar(fields, "utility", float),
                prior=_scalar(fields, "prior", float),
                lcb=_scalar(fields, "lcb", float),
                utility_lcb=_scalar(fields, "utilityLcb", float),
                pv=pv,
                weight=_scalar(fields, "weight", float),
                order=_scalar(fields, "order", int),
            )
        )
    play_matches = re.findall(r"(?m)^play\s+(\S+)\s*$", response.payload)
    if not play_matches:
        raise GTPError("completed kata-genmove_analyze response has no final play record")
    if not candidates:
        raise GTPError("completed kata-genmove_analyze response has no candidates")
    return CompletedAnalysis(tuple(candidates), play_matches[-1].lower(), response.raw)


class GTPClient:
    """Single-flight GTP client with byte buffering and command IDs."""

    def __init__(
        self,
        argv: Iterable[str],
        *,
        log_directory: Path,
        default_timeout: float = 30.0,
    ) -> None:
        self.argv = tuple(argv)
        self.default_timeout = default_timeout
        log_directory.mkdir(parents=True, exist_ok=True)
        self._stdin_log = (log_directory / "stdin.raw").open("wb")
        self._stdout_log = (log_directory / "stdout.raw").open("wb")
        self._stderr_log = (log_directory / "stderr.raw").open("wb")
        self.process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        os.set_blocking(self.process.stdout.fileno(), False)
        os.set_blocking(self.process.stderr.fileno(), False)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ, "stdout")
        self._selector.register(self.process.stderr, selectors.EVENT_READ, "stderr")
        self._stdout_buffer = bytearray()
        self._stdout_tail = bytearray()
        self._stderr_tail = bytearray()
        self._pending: dict[int | None, list[GTPResponse]] = {}
        self._next_id = 1
        self._closed = False
        self.started_monotonic = time.monotonic()

    def __enter__(self) -> "GTPClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def command(
        self,
        command: str,
        *,
        timeout: float | None = None,
        raise_on_error: bool = True,
    ) -> GTPResponse:
        if self._closed or self.process.poll() is not None:
            raise GTPError("GTP process is not running")
        command_id = self._next_id
        self._next_id += 1
        wire = f"{command_id} {command}\n".encode("utf-8")
        self._stdin_log.write(wire)
        self._stdin_log.flush()
        assert self.process.stdin is not None
        self.process.stdin.write(wire)
        self.process.stdin.flush()
        try:
            response = self._wait_for(command_id, timeout or self.default_timeout)
        except BrokenPipeError as error:
            raise GTPError(f"failed to write GTP command {command_id}: {self.diagnostics()}") from error
        if raise_on_error and not response.success:
            raise GTPProtocolError(response)
        return response

    def _wait_for(self, command_id: int, timeout: float) -> GTPResponse:
        deadline = time.monotonic() + timeout
        while True:
            queued = self._pending.get(command_id)
            if queued:
                return queued.pop(0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout(command_id, timeout)
            events = self._selector.select(remaining)
            if not events:
                raise self._timeout(command_id, timeout)
            for key, _ in events:
                stream: BinaryIO = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    try:
                        self._selector.unregister(stream)
                    except KeyError:
                        pass
                    continue
                if key.data == "stdout":
                    self._stdout_log.write(chunk)
                    self._stdout_log.flush()
                    self._append_tail(self._stdout_tail, chunk)
                    self._stdout_buffer.extend(chunk)
                    self._extract_complete_frames()
                else:
                    self._stderr_log.write(chunk)
                    self._stderr_log.flush()
                    self._append_tail(self._stderr_tail, chunk)
            if self.process.poll() is not None:
                self._drain_ready_streams()
                raise GTPError(
                    f"GTP process exited while waiting for command {command_id}: {self.diagnostics()}"
                )

    @staticmethod
    def _append_tail(buffer: bytearray, chunk: bytes, limit: int = 4096) -> None:
        buffer.extend(chunk)
        if len(buffer) > limit:
            del buffer[:-limit]

    def _drain_ready_streams(self) -> None:
        """Capture already-buffered child diagnostics before surfacing an exit."""

        for key, _ in self._selector.select(0):
            stream: BinaryIO = key.fileobj
            try:
                chunk = os.read(stream.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            if key.data == "stdout":
                self._stdout_log.write(chunk)
                self._stdout_log.flush()
                self._append_tail(self._stdout_tail, chunk)
                self._stdout_buffer.extend(chunk)
                self._extract_complete_frames()
            else:
                self._stderr_log.write(chunk)
                self._stderr_log.flush()
                self._append_tail(self._stderr_tail, chunk)

    def diagnostics(self) -> dict[str, object]:
        """Small, serializable process/protocol snapshot for failure manifests."""

        return {
            "pid": self.process.pid,
            "poll": self.process.poll(),
            "returncode": self.process.returncode,
            "elapsed_since_start_seconds": time.monotonic() - self.started_monotonic,
            "next_command_id": self._next_id,
            "stdout_tail": self._stdout_tail.decode("utf-8", errors="replace"),
            "stderr_tail": self._stderr_tail.decode("utf-8", errors="replace"),
            "closed": self._closed,
        }

    def _timeout(self, command_id: int, timeout: float) -> GTPTimeout:
        self._drain_ready_streams()
        return GTPTimeout(
            f"timed out waiting for GTP command {command_id} after {timeout:.3f}s: "
            f"{self.diagnostics()}"
        )

    def _extract_complete_frames(self) -> None:
        while True:
            match = _FRAME_START.search(self._stdout_buffer)
            if match is None:
                return
            frame_start = match.start(1)
            frame_end = self._stdout_buffer.find(b"\n\n", frame_start)
            if frame_end < 0:
                # Drop non-protocol startup chatter only after locating a frame.
                if frame_start > 0:
                    del self._stdout_buffer[:frame_start]
                return
            frame_end += 2
            frame = bytes(self._stdout_buffer[frame_start:frame_end])
            del self._stdout_buffer[:frame_end]
            response = self._parse_frame(frame)
            self._pending.setdefault(response.command_id, []).append(response)

    @staticmethod
    def _parse_frame(frame: bytes) -> GTPResponse:
        raw = frame.decode("utf-8", errors="replace")
        lines = raw[:-2].split("\n")
        header = re.fullmatch(r"([=?])(\d*) ?(.*)", lines[0])
        if header is None:
            raise GTPError(f"malformed GTP response header: {lines[0]!r}")
        marker, command_id_text, first_payload = header.groups()
        payload_lines = ([first_payload] if first_payload else []) + lines[1:]
        return GTPResponse(
            command_id=int(command_id_text) if command_id_text else None,
            success=marker == "=",
            payload="\n".join(payload_lines),
            raw=raw,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.command("quit", timeout=5.0, raise_on_error=False)
                except GTPError:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        finally:
            self._closed = True
            self._selector.close()
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            for handle in (self._stdin_log, self._stdout_log, self._stderr_log):
                handle.close()
