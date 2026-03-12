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
