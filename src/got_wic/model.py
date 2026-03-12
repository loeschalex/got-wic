from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Objective:
    name: str
    count: int
    first_capture: int
    hold_pts_min: int
    opens_at: int
    zone: str


@dataclass(frozen=True)
class Dragon:
    name: str
    escort_pts: int
    spawns_at: int


@dataclass(frozen=True)
class TreasureConfig:
    starts_at: int = 8
    normal_pts: int = 80
    rare_pts: int = 120


@dataclass(frozen=True)
class GameConfig:
    objectives: list[Objective]
    dragons: list[Dragon]
    treasure: TreasureConfig = field(default_factory=TreasureConfig)
    match_duration: int = 60
    phase_boundaries: list[int] = field(default_factory=lambda: [0, 8, 12])


@dataclass
class Allocation:
    assignments: dict[str, dict[str, int]]  # phase_name -> {objective_name -> armies}
    total_armies: int

    def armies_used(self, phase: str) -> int:
        return sum(self.assignments.get(phase, {}).values())

    def armies_unused(self, phase: str) -> int:
        return self.total_armies - self.armies_used(phase)

    def is_valid(self, phase: str) -> bool:
        return self.armies_used(phase) <= self.total_armies


@dataclass(frozen=True)
class PlayerTier:
    name: str
    combat_power: float
    healing_capacity: int  # -1 = unlimited


@dataclass(frozen=True)
class AllianceProfile:
    tiers: list[PlayerTier]
    counts: list[int]

    def __post_init__(self):
        if len(self.tiers) != len(self.counts):
            raise ValueError(
                f"tiers ({len(self.tiers)}) and counts ({len(self.counts)}) must have same length"
            )

    @property
    def total_players(self) -> int:
        return sum(self.counts)

    @property
    def total_power(self) -> float:
        return sum(t.combat_power * c for t, c in zip(self.tiers, self.counts))


def default_alliance_profile(total_players: int) -> AllianceProfile:
    """Create a default 4-tier alliance profile for a given player count.

    Distributes players roughly: 3% whale, 17% dolphin, 30% minnow, 50% alt.
    """
    n_whale = max(1, round(total_players * 0.03))
    n_dolphin = round(total_players * 0.17)
    n_minnow = round(total_players * 0.30)
    n_alt = total_players - n_whale - n_dolphin - n_minnow
    tiers = [
        PlayerTier("whale", 100.0, -1),
        PlayerTier("dolphin", 30.0, 8),
        PlayerTier("minnow", 8.0, 4),
        PlayerTier("alt", 1.0, 2),
    ]
    return AllianceProfile(tiers=tiers, counts=[n_whale, n_dolphin, n_minnow, n_alt])


def default_config() -> GameConfig:
    objectives = [
        Objective("Stark Outpost", 2, 200, 80, 0, "near_stark"),
        Objective("Greyjoy Outpost", 2, 200, 80, 0, "near_greyjoy"),
        Objective("Armory", 1, 400, 120, 0, "center"),
        Objective("Hot Spring", 1, 400, 120, 0, "center"),
        Objective("Stronghold", 4, 600, 180, 12, "mid"),
    ]
    dragons = [
        Dragon("Winter Ice", 3000, 12),
        Dragon("Flaming Sun", 3000, 12),
    ]
    return GameConfig(objectives=objectives, dragons=dragons)
