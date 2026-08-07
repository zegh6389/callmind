from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .codec import dbfs


class VADEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADEvent:
    type: VADEventType
    frame_index: int


class EnergyVAD:
    def __init__(
        self,
        energy_dbfs: float = -45.0,
        start_frames: int = 3,
        end_frames: int = 25,
    ) -> None:
        self.energy_dbfs = energy_dbfs
        self.start_frames = start_frames
        self.end_frames = end_frames
        self._active_run = 0
        self._silent_run = 0
        self._in_speech = False
        self._frame_index = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._active_run = 0
        self._silent_run = 0
        self._in_speech = False

    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        self._frame_index += 1
        active = dbfs(frame) >= self.energy_dbfs
        event: VADEvent | None = None

        if not self._in_speech:
            if active:
                self._active_run += 1
                if self._active_run >= self.start_frames:
                    self._in_speech = True
                    self._silent_run = 0
                    event = VADEvent(VADEventType.SPEECH_START, self._frame_index)
            else:
                self._active_run = 0
        else:
            if active:
                self._silent_run = 0
            else:
                self._silent_run += 1
                if self._silent_run >= self.end_frames:
                    self._in_speech = False
                    self._active_run = 0
                    event = VADEvent(VADEventType.SPEECH_END, self._frame_index)

        return event
