from pathlib import Path

import numpy as np
import soundfile as sf


class PredictionModel:
    batch_size: int = 10

    def __init__(self) -> None:
        pass

    def predict(self, batch: list[Path]) -> list[float]:
        """Получает батч путей к WAV-файлам, возвращает список anomaly_score."""
        scores = []
        for path in batch:
            try:
                audio, _ = sf.read(str(path), always_2d=True)
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            except Exception:
                rms = 0.0
            scores.append(rms)
        return scores
