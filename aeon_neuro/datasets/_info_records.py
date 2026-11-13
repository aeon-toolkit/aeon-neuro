"""Named internal metadata records and their portable dictionary boundary."""

from copy import deepcopy
from dataclasses import dataclass, field, fields


@dataclass(kw_only=True)
class _Record:
    """Preserve supplied keys and extensions when restoring portable records."""

    # Omitted optional fields must stay omitted in saved state. Defaults support
    # attribute access without adding invented source facts to the wire format.
    _present: frozenset[str] = field(default_factory=frozenset, repr=False)
    _extra: dict = field(default_factory=dict, repr=False)

    @classmethod
    def _from_dict(cls, value):
        """Detach named fields and retain unrecognized source extensions."""
        names = {item.name for item in fields(cls) if not item.name.startswith("_")}
        return cls(
            **{key: deepcopy(v) for key, v in value.items() if key in names},
            _present=frozenset(value),
            _extra={key: deepcopy(v) for key, v in value.items() if key not in names},
        )

    def to_dict(self):
        """Return detached portable values with original optional-key presence."""
        result = deepcopy(self._extra)
        for item in fields(self):
            if item.name not in self._present or item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            result[item.name] = (
                [v.to_dict() if isinstance(v, _Record) else deepcopy(v) for v in value]
                if isinstance(value, list)
                else deepcopy(value)
            )
        return result


@dataclass
class _DatasetFacts(_Record):
    """Original dataset facts, separate from current case membership."""

    selected_participants: list[str | int] = field(default_factory=list)
    description: dict | None = None
    readme: str | None = None
    participants: dict | None = None
    participants_json: dict | None = None


@dataclass
class _RecordingDescriptor(_Record):
    """Original recording identity and layout with optional source annotations."""

    recording_id: str
    participant_id: str | int | None
    channel_names: list[str]
    sampling_frequency: float
    n_samples: int
    subject: str | None = None
    session: str | None = None
    task: str | None = None
    acq: str | None = None
    run: str | None = None
    channel_metadata: list[dict] | None = None
    channel_units_fiff: list[int] = field(default_factory=list)
    eeg_metadata: dict = field(default_factory=dict)
    electrodes: dict | None = None
    coordsystem: dict | None = None
    scan: dict | None = None


@dataclass
class _CaseRow(_Record):
    """One current row's identity and original contributing sample support."""

    case_id: str
    recording_id: str | None
    participant_id: str | int | None
    group_id: str | int | None
    source_support: list[int]


@dataclass
class _SignalLayout(_Record):
    """Current signal layout and its mapping onto an optional source grid."""

    channel_names: list[str]
    sampling_frequency: float
    units: list[str | None]
    n_samples: int
    source_origin_sample: float
    source_sample_step: float
    reference: object = None
    temporal_context: str = "whole_row"
    source_channel_names: list[str] | None = None


@dataclass
class _CurrentView(_Record):
    """Aligned current rows and layouts, separate from source context."""

    version: int
    kind: str
    rows: list[_CaseRow]
    layouts: list[_SignalLayout]

    @classmethod
    def _from_dict(cls, value):
        """Restore typed children after portable-state validation."""
        return cls(
            version=value["version"],
            kind=value["kind"],
            rows=[_CaseRow._from_dict(row) for row in value["rows"]],
            layouts=[_SignalLayout._from_dict(layout) for layout in value["layouts"]],
            _present=frozenset(value),
            _extra={
                key: deepcopy(v)
                for key, v in value.items()
                if key not in {"version", "kind", "rows", "layouts"}
            },
        )
