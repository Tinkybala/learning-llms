import numpy as np
from numpy.typing import NDArray
from sentence_transformers.cross_encoder import CrossEncoder

from models.base import SingletonMeta
from settings import settings


class CrossEncoderModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID,
        device: str = settings.RAG_MODEL_DEVICE,
    ) -> None:
        """
        Sinlgeton class that provides a pre-trained cross-encoder for reranking
        """
        self._model_id = model_id
        self._device = device

        self._model = CrossEncoder(model_name=self._model_id, device=self._device)
        self._model.model.eval()

    def __call__(
        self, pairs: list[tuple[str, str]], to_list: bool = True
    ) -> NDArray[np.float32 | list[float]]:
        scores = self._model.predict(pairs)

        if to_list:
            scores = scores.tolist()

        return scores
