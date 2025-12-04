"""
Entidade Track do domínio.
"""

from typing import List, Dict, Any, Optional
from domain.value_objects import IdVO
from domain.entities.event_entity import Event


class Track:
    """
    Entidade que representa um track (rastreamento) de uma face ao longo de múltiplos frames.
    Um track contém uma lista de eventos (detecções) da mesma face em diferentes frames.
    """

    def __init__(
        self,
        id: IdVO,
        events: Optional[List[Event]] = None
    ):
        """
        Inicializa a entidade Track.

        :param id: ID único do track (IdVO).
        :param events: Lista de eventos do track (opcional, default=[]).
        :raises TypeError: Se algum parâmetro não for do tipo esperado.
        """
        if not isinstance(id, IdVO):
            raise TypeError(f"id deve ser IdVO, recebido: {type(id).__name__}")
        
        if events is not None and not isinstance(events, list):
            raise TypeError(f"events deve ser lista, recebido: {type(events).__name__}")
        
        self._id = id
        self._events: List[Event] = []
        
        if events is not None:
            for event in events:
                self.add_event(event)

    @property
    def id(self) -> IdVO:
        """Retorna o ID do track."""
        return self._id

    @property
    def events(self) -> List[Event]:
        """Retorna a lista de eventos do track (cópia para evitar modificações externas)."""
        return self._events.copy()

    @property
    def event_count(self) -> int:
        """Retorna a quantidade de eventos no track."""
        return len(self._events)

    @property
    def is_empty(self) -> bool:
        """Verifica se o track está vazio (sem eventos)."""
        return len(self._events) == 0

    def add_event(self, event: Event) -> None:
        """
        Adiciona um evento ao track.

        :param event: Evento a ser adicionado.
        :raises TypeError: Se event não for do tipo Event.
        """
        if not isinstance(event, Event):
            raise TypeError(f"event deve ser Event, recebido: {type(event).__name__}")
        
        self._events.append(event)

    def get_best_event(self) -> Optional[Event]:
        """
        Retorna o evento com o maior score de qualidade facial.

        :return: Evento com melhor qualidade ou None se track estiver vazio.
        """
        if self.is_empty:
            return None
        
        return max(self._events, key=lambda e: e.face_quality_score.value())

    def get_first_event(self) -> Optional[Event]:
        """
        Retorna o primeiro evento do track.

        :return: Primeiro evento ou None se track estiver vazio.
        """
        if self.is_empty:
            return None
        return self._events[0]

    def get_last_event(self) -> Optional[Event]:
        """
        Retorna o último evento do track.

        :return: Último evento ou None se track estiver vazio.
        """
        if self.is_empty:
            return None
        return self._events[-1]

    def get_average_confidence(self) -> float:
        """
        Calcula a confiança média das detecções no track.

        :return: Confiança média ou 0.0 se track estiver vazio.
        """
        if self.is_empty:
            return 0.0
        
        total_confidence = sum(event.confidence.value() for event in self._events)
        return total_confidence / len(self._events)

    def get_average_quality_score(self) -> float:
        """
        Calcula o score médio de qualidade facial no track.

        :return: Score médio de qualidade ou 0.0 se track estiver vazio.
        """
        if self.is_empty:
            return 0.0
        
        total_quality = sum(event.face_quality_score.value() for event in self._events)
        return total_quality / len(self._events)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a entidade para um dicionário.

        :return: Dicionário com os dados do track.
        """
        return {
            'id': self._id.value(),
            'event_count': self.event_count,
            'events': [event.to_dict() for event in self._events],
            'average_confidence': self.get_average_confidence(),
            'average_quality_score': self.get_average_quality_score(),
            'best_event_id': self.get_best_event().id.value() if not self.is_empty else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Track':
        """
        Cria uma instância de Track a partir de um dicionário.
        Nota: Este método não reconstrói os eventos, apenas o ID do track.
        Para reconstruir eventos, eles devem ser adicionados manualmente.

        :param data: Dicionário com os dados do track.
        :return: Instância de Track.
        :raises KeyError: Se alguma chave obrigatória estiver ausente.
        """
        return cls(
            id=IdVO(data['id']),
            events=None
        )

    def __eq__(self, other) -> bool:
        """Compara dois tracks por igualdade (baseado no ID)."""
        if not isinstance(other, Track):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        """Retorna o hash do track (baseado no ID)."""
        return hash(self._id)

    def __repr__(self) -> str:
        """Representação string do track."""
        best_quality = self.get_best_event().face_quality_score.value() if not self.is_empty else 0.0
        return (
            f"Track(id={self._id.value()}, "
            f"events={self.event_count}, "
            f"avg_quality={self.get_average_quality_score():.4f}, "
            f"best_quality={best_quality:.4f})"
        )

    def __str__(self) -> str:
        """Conversão para string."""
        return (
            f"Track {self._id.value()}: "
            f"{self.event_count} events, "
            f"avg quality {self.get_average_quality_score():.4f}"
        )
