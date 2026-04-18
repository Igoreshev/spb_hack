import joblib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

FEATURE_SIZE = 42


def _aggregate(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 1:
        feature = feature[np.newaxis, :]
    return np.hstack([np.mean(feature, axis=1), np.std(feature, axis=1)])


def _anomaly_score(model, X: np.ndarray) -> np.ndarray:
    try:
        if hasattr(model, "decision_function"):
            scores = -model.decision_function(X)
        elif hasattr(model, "score_samples"):
            scores = -model.score_samples(X)
        else:
            scores = np.zeros(X.shape[0])
        # Очистка NaN/inf
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        return scores
    except Exception:
        return np.zeros(X.shape[0])


def _extract_features_from_audio(wav_path: Path) -> np.ndarray:
    try:
        # sr частота дискретизации, y - количество семплов
        y, sr = librosa.load(str(wav_path), sr=48000, mono=True) # попробовать с другой частотой
        if y.size == 0:
            raise ValueError("Empty audio")
        
        # гармоника апмлитуда на кратных частотах, апмлитуда сигнала которая берется на частотах на кратной частоте 
        y_harmonic, y_percussive = librosa.effects.hpss(y) # след итерацию попробовать без этого
        #тон и перкуссия
        #средний коэффициент гаромоники к перкуссии
        harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_percussive)) + 1e-9)
        

        # СПЕКТРАЛЬНЫЕ ПРИЗНАКИ
        # 20 коэффициентов описывают "форму" звука (паспорт звука, MFCC - мел-частотные кепстральные коэффициенты)/ характеризует трембр звука

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # поменять коэф на 20 с ним будет дольше но лучше работать
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16, fmax=24000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        

        # Плоскость спектра (отлично ловит посторонние свисты)
        flatness = librosa.feature.spectral_flatness(y=y)

        # Спектральный центроид: "яркость" звука (свист/гул)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        # Спектральный спад: помогает отличить шум от чистого тона
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)


        # ВРЕМЕННЫЕ ПРИЗНАКИ
        # ZCR: частота пересечения нуля (помогает найти трение/скрежет)
        zcr = librosa.feature.zero_crossing_rate(y)
        # RMS: энергия/громкость
        rms = librosa.feature.rms(y=y)


        feature_vector = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(mel_db, axis=1),
            np.mean(centroid),
            np.mean(rolloff),
            np.mean(zcr),
            np.mean(flatness),
            np.mean(rms), # также думаю вместо средного использвать моду, может, будет лучший скор
            harmonic_ratio, # попробовать добавить персентиль, с маленький доверительный интервалом
        ])

        return feature_vector.astype(np.float32)
    except Exception:
        return np.zeros(42, dtype=np.float32)



class PredictionModel:
    batch_size: int = 16

    def __init__(self) -> None:
        base_path = Path(__file__).parent
        best_model_path = base_path / "best_model.pkl"
        fallback_model_path = base_path / "model.pkl"
        scaler_path = base_path / "scaler.pkl"

        self.model = None
        self.scaler = None
        self.pca = None
        self.score_center = 0.0
        self.score_scale = 1.0

        model_path = best_model_path if best_model_path.exists() else fallback_model_path
        if model_path.exists():
            model_obj = joblib.load(model_path)
            if isinstance(model_obj, dict):
                if "model" in model_obj:
                    self.model = model_obj["model"]
                elif "models" in model_obj and len(model_obj["models"]) > 0:
                    self.model = model_obj["models"][0]
            else:
                self.model = model_obj
        else:
            self.model = None
            print("WARNING: best_model.pkl or model.pkl not found!")

        if scaler_path.exists():
            scaler_obj = joblib.load(scaler_path)
            if isinstance(scaler_obj, dict):
                self.scaler = scaler_obj.get("scaler")
                self.pca = scaler_obj.get("pca")
                self.score_center = scaler_obj.get("score_center", 0.0)
                self.score_scale = scaler_obj.get("score_scale", 1.0)
            else:
                self.scaler = scaler_obj
        else:
            self.scaler = None
            print("WARNING: scaler.pkl not found!")

    def _extract_features(self, wav_path: Path) -> np.ndarray:
        return _extract_features_from_audio(wav_path)

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(X.shape[0], dtype=np.float32)

        scores = _anomaly_score(self.model, X)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        return scores.astype(np.float32)

    def predict(self, batch: list[Path]) -> list[float]:
        if len(batch) == 0:
            return []

        features_batch = [self._extract_features(path) for path in batch]
        X = np.array(features_batch, dtype=np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.pca is not None:
            X = self.pca.transform(X)

        raw_scores = self._score(X)
        normalized = (raw_scores - self.score_center) / (self.score_scale + 1e-9)
        # Очистка NaN/inf перед возвратом
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return normalized.astype(float).tolist()


def train_and_save(
    manifest_path: Path,
    audio_root: Path,
    model_path: Path,
    scaler_path: Path,
    contamination: float = 0.02,
) -> None:
    df = pd.read_csv(manifest_path)
    feature_list = []
    for filename in df["filename"]:
        path = audio_root / filename
        feature_list.append(_extract_features_from_audio(path))

    X = np.array(feature_list, dtype=np.float32)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.98, svd_solver="full", random_state=42)
    X_proj = pca.fit_transform(X_scaled)

    iforest = IsolationForest(
        n_estimators=256,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_proj)

    model_dict = {
        "model": iforest,
    }

    score_vector = _anomaly_score(iforest, X_proj)
    score_scale = float(np.std(score_vector))
    if score_scale < 1e-6:
        score_scale = 1.0
    scaler_dict = {
        "scaler": scaler,
        "pca": pca,
        "score_center": float(np.mean(score_vector)),
        "score_scale": score_scale,
    }

    joblib.dump(model_dict, model_path)
    joblib.dump(model_dict, model_path.with_name("best_model.pkl"))
    joblib.dump(scaler_dict, scaler_path)
    print(f"Saved model to {model_path.name}, best_model.pkl, and scaler to {scaler_path.name}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data_root = root.parent / "train"
    manifest = data_root / "train.csv"
    audio_dir = data_root / "train"
    train_and_save(
        manifest_path=manifest,
        audio_root=audio_dir,
        model_path=root / "model.pkl",
        scaler_path=root / "scaler.pkl",
    )


 
