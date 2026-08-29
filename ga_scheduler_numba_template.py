import random
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from numba import njit

RANDOM_SEED = 42
POP_SIZE = 500
GENERATIONS = 175
OPTIMIZATION_RUNS = 10
TRAIN_COUNT = 50
MIN_CURRENT_DELAY = 20
MAX_CURRENT_DELAY = 200
MUTATION_RATE = 0.25
ELITISM_RATIO = 0.2
STAGNATION_LIMIT = 20
ALGORITHM_VERSION = "numba-ga-scheduler-v1"
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
PRIORITY_WEIGHTS = {"express": 3, "passenger": 2, "freight": 1}
MIN_HEADWAY = 2


@dataclass(frozen=True)
class Train:
    train_id: str
    train_type: str
    arrival: int
    cross_time: int
    platform: int
    priority: Optional[int] = None
    delay_minutes: int = 0


@dataclass(frozen=True)
class SchedulerSnapshot:
    snapshot_id: str
    trains: Tuple[Train, ...]
    minimum_headway_minutes: int = MIN_HEADWAY
    maintenance_windows: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ScheduleEntry:
    train_id: str
    sequence: int
    platform: int
    planned_start: int
    planned_end: int
    delay_minutes: int
    weighted_delay: int
    current_delay_minutes: int = 0
    total_delay_minutes: int = 0


@dataclass(frozen=True)
class OptimizationResult:
    optimization_id: str
    snapshot_id: str
    status: str
    objective_score: float
    schedule: Tuple[ScheduleEntry, ...]
    metrics: Dict[str, float]
    generations_completed: int
    random_seed: int
    algorithm_version: str = ALGORITHM_VERSION

    def as_dict(self):

        result = asdict(self)
        result["schedule"] = [asdict(x) for x in self.schedule]
        return result


def validate_snapshot(snapshot):
    if not snapshot.snapshot_id:
        raise ValueError("snapshot id missing")

    ids = [t.train_id for t in snapshot.trains]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate train IDs")
    if snapshot.minimum_headway_minutes < 0:
        raise ValueError("minimum headway cannot be negative")
    for train in snapshot.trains:
        if not train.train_id:
            raise ValueError("train ID missing")
        if train.train_type not in PRIORITY_WEIGHTS:
            raise ValueError(f"unsupported train type: {train.train_type}")
        if train.arrival < 0 or train.cross_time <= 0 or train.platform <= 0:
            raise ValueError(f"invalid train data for {train.train_id}")
        if train.delay_minutes < 0:
            raise ValueError(f"delay cannot be negative for {train.train_id}")
    for start, end in snapshot.maintenance_windows:
        if start >= end:
            raise ValueError("maintenance window must have start < end")


def snapshot_train_data(snapshot):
    return {
        train.train_id: {
            "type": train.train_type,
            "arrival": train.arrival,
            "effective_arrival": train.arrival + train.delay_minutes,
            "cross_time": train.cross_time,
            "platform": train.platform,
            "priority": train.priority,
            "delay_minutes": train.delay_minutes,
        }
        for train in snapshot.trains
    }


def prepare_numba_data(trains):

    train_ids = list(trains.keys())

    arrivals = []

    crosses = []

    platforms = []

    weights = []

    for tid in train_ids:

        t = trains[tid]

        arrivals.append(t["effective_arrival"])

        crosses.append(t["cross_time"])

        platforms.append(t["platform"])

        weight = (
            t["priority"] if t["priority"] is not None else PRIORITY_WEIGHTS[t["type"]]
        )

        weights.append(weight)

    return (
        train_ids,
        np.array(arrivals, dtype=np.int32),
        np.array(crosses, dtype=np.int32),
        np.array(platforms, dtype=np.int32),
        np.array(weights, dtype=np.int32),
    )


@njit(cache=True)
def evaluate_schedule_numba(
    schedule, arrivals, crosses, platforms, weights, minimum_headway
):

    platform_free = np.zeros(100, dtype=np.int32)

    score = 0

    for i in range(schedule.shape[0]):

        idx = schedule[i]

        platform = platforms[idx]

        arrival = arrivals[idx]

        cross = crosses[idx]

        start = arrival

        free_time = platform_free[platform] + minimum_headway

        if free_time > start:

            start = free_time

        delay = start - arrival

        if delay > 0:

            score += delay * weights[idx]

        platform_free[platform] = start + cross

    return score


_WORKER_DATA = None


def init_worker(data):

    global _WORKER_DATA

    _WORKER_DATA = data


def fitness_worker(schedule):

    return evaluate_schedule_numba(np.asarray(schedule, dtype=np.int32), *_WORKER_DATA)


def parallel_fitness(population, numba_data, minimum_headway=MIN_HEADWAY):

    # numba_data:
    # (
    # train_ids,
    # arrivals,
    # crosses,
    # platforms,
    # weights
    # )

    data = numba_data[1:]

    scores = []

    for chromosome in population:

        score = evaluate_schedule_numba(
            np.asarray(chromosome, dtype=np.int32),
            data[0],  # arrivals
            data[1],  # crosses
            data[2],  # platforms
            data[3],  # weights
            minimum_headway,
        )

        scores.append(score)

    return scores


def create_random_individual_fast(train_count):

    return np.random.permutation(train_count).astype(np.int32)


def order_crossover_fast(parent1, parent2):

    size = len(parent1)

    a, b = sorted(random.sample(range(size), 2))

    child = np.full(size, -1, dtype=np.int32)

    # copy part of parent 1

    child[a:b] = parent1[a:b]

    used = set(child[child != -1])

    index = b % size

    for gene in parent2:

        if gene not in used:

            while child[index] != -1:

                index = (index + 1) % size

            child[index] = gene

            used.add(gene)

    return child


def swap_mutation_fast(chromosome):

    a, b = random.sample(range(len(chromosome)), 2)

    chromosome[a], chromosome[b] = (chromosome[b], chromosome[a])


def tournament_selection_fast(population, scores, k=3):

    selected = random.sample(range(len(population)), k)

    best = selected[0]

    for idx in selected:

        if scores[idx] < scores[best]:

            best = idx

    return population[best]


def run_ga_fast(
    numba_data,
    generations,
    pop_size,
    mutation_rate,
    stagnation_limit,
    minimum_headway=MIN_HEADWAY,
):

    train_count = len(numba_data[0])

    # --------------------------
    # Initial population
    # --------------------------

    population = [create_random_individual_fast(train_count) for _ in range(pop_size)]

    best_score = float("inf")

    stagnant = 0

    completed = 0

    for gen in range(generations):

        completed += 1

        # ----------------------
        # FITNESS
        # ----------------------

        scores = parallel_fitness(population, numba_data, minimum_headway)

        ranked = sorted(zip(scores, population), key=lambda x: x[0])

        scores = [x[0] for x in ranked]

        population = [x[1] for x in ranked]

        current_best = scores[0]

        if current_best < best_score:

            best_score = current_best

            stagnant = 0

        else:

            stagnant += 1

        if gen % 10 == 0:

            print(f"[GA] Generation {gen}/{generations} " f"Score={current_best}")

        if stagnant >= stagnation_limit:

            break

        # ----------------------
        # ELITISM
        # ----------------------

        elite_count = max(1, int(pop_size * ELITISM_RATIO))

        new_population = [population[i].copy() for i in range(elite_count)]

        # ----------------------
        # CREATE CHILDREN
        # ----------------------

        while len(new_population) < pop_size:

            parent1 = tournament_selection_fast(population, scores)

            parent2 = tournament_selection_fast(population, scores)

            child = order_crossover_fast(parent1, parent2)

            if random.random() < mutation_rate:

                swap_mutation_fast(child)

            new_population.append(child)

        population = new_population

    # final best chromosome

    final_scores = parallel_fitness(population, numba_data, minimum_headway)

    best_index = int(np.argmin(final_scores))

    return (population[best_index], final_scores[best_index], completed)


def build_schedule_entries(schedule, snapshot):

    train_data = snapshot_train_data(snapshot)

    platform_free = {}

    entries = []

    for seq, tid in enumerate(schedule, start=1):

        train = train_data[tid]

        platform = train["platform"]

        arrival = train["effective_arrival"]

        cross = train["cross_time"]

        start = max(
            platform_free.get(platform, 0) + snapshot.minimum_headway_minutes, arrival
        )

        end = start + cross

        delay = max(0, start - arrival)

        weight = (
            train["priority"]
            if train["priority"] is not None
            else PRIORITY_WEIGHTS[train["type"]]
        )

        entries.append(
            ScheduleEntry(
                train_id=tid,
                sequence=seq,
                platform=platform,
                planned_start=start,
                planned_end=end,
                delay_minutes=delay,
                weighted_delay=delay * weight,
                current_delay_minutes=train["delay_minutes"],
                total_delay_minutes=train["delay_minutes"] + delay,
            )
        )

        platform_free[platform] = end

    return tuple(entries)


def print_schedule(schedule, trains):

    platform_free = {}

    total = 0

    print("=" * 80)

    print(
        "Train | Type | Planned | Current | Effective | Start | End | New delay | Total delay | Platform"
    )

    print("-" * 80)

    for tid in schedule:

        t = trains[tid]

        platform = t["platform"]

        effective_arrival = t["effective_arrival"]
        start = max(platform_free.get(platform, 0) + MIN_HEADWAY, effective_arrival)

        end = start + t["cross_time"]

        delay = max(0, start - effective_arrival)
        total_delay = t["delay_minutes"] + delay

        total += delay

        platform_free[platform] = end

        print(
            f"{tid:5} | "
            f"{t['type']:9} | "
            f"{t['arrival']:7} | "
            f"{t['delay_minutes']:7} | "
            f"{effective_arrival:9} | "
            f"{start:5} | "
            f"{end:3} | "
            f"{delay:9} | "
            f"{total_delay:11} | "
            f"{platform}"
        )

    print("-" * 80)

    print("Total new delay:", total)

    print("=" * 80)


class SchedulerService:

    def __init__(self, trains, maintenance_windows=None):

        self.trains = trains

        self.maintenance_windows = maintenance_windows or []

    def optimize(self, snapshot, runs=OPTIMIZATION_RUNS):

        validate_snapshot(snapshot)

        print(f"[Service] Starting {runs} GA runs...")

        # convert data

        trains = snapshot_train_data(snapshot)

        numba_data = prepare_numba_data(trains)

        # ====================================================
        # RUN MULTIPLE GA INSTANCES
        # ====================================================

        args = [
            (
                numba_data,
                GENERATIONS,
                POP_SIZE,
                MUTATION_RATE,
                STAGNATION_LIMIT,
                snapshot.minimum_headway_minutes,
            )
            for _ in range(runs)
        ]

        with mp.Pool(processes=min(runs, mp.cpu_count() - 1)) as pool:

            results = pool.starmap(
                run_ga_fast,
                [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in args],
            )

        # ====================================================
        # SELECT BEST RESULT
        # ====================================================

        best = None

        best_score = float("inf")

        best_generation = 0

        for i, result in enumerate(results):

            chromosome, score, generation = result

            print(f"[Service] Run {i+1}/{runs} " f"Score={score}")

            if score < best_score:

                best_score = score

                best = chromosome

                best_generation = generation

        # ====================================================
        # CONVERT INTEGER ARRAY BACK TO TRAIN IDS
        # ====================================================

        train_ids = numba_data[0]

        final_schedule = [train_ids[x] for x in best]

        entries = build_schedule_entries(final_schedule, snapshot)

        new_delay = sum(e.delay_minutes for e in entries)
        current_delay = sum(e.current_delay_minutes for e in entries)
        total_delay = current_delay + new_delay
        weighted_delay = sum(e.weighted_delay for e in entries)

        return OptimizationResult(
            optimization_id="opt_" + snapshot.snapshot_id,
            snapshot_id=snapshot.snapshot_id,
            status="completed",
            objective_score=float(best_score),
            schedule=entries,
            metrics={
                "total_delay": total_delay,
                "current_delay": current_delay,
                "new_delay": new_delay,
                "weighted_delay": weighted_delay,
                "average_delay": total_delay / len(entries) if entries else 0,
                "maximum_delay": max([e.delay_minutes for e in entries], default=0),
                "trains_scheduled": len(entries),
            },
            generations_completed=best_generation,
            random_seed=RANDOM_SEED,
        )


def build_combined_scenario():
    rng = random.Random(RANDOM_SEED)

    train_types = ("express", "passenger", "passenger", "freight")

    trains = []

    for i in range(1, TRAIN_COUNT + 1):

        train_type = rng.choice(train_types)

        trains.append(
            Train(
                train_id=f"T{i}",
                train_type=train_type,
                arrival=rng.randint(600, 660),
                cross_time=rng.randint(2, 6),
                platform=rng.randint(1, 24),
                delay_minutes=rng.randint(MIN_CURRENT_DELAY, MAX_CURRENT_DELAY),
            )
        )

    maintenance_start = rng.randint(620, 640)

    return SchedulerSnapshot(
        snapshot_id="demo",
        trains=tuple(trains),
        minimum_headway_minutes=MIN_HEADWAY,
        maintenance_windows=((maintenance_start, maintenance_start + 5),),
    )


def plot_schedule(snapshot, result, output="train_schedule.png"):

    train_types = {t.train_id: t.train_type for t in snapshot.trains}

    platforms = sorted(set(e.platform for e in result.schedule))

    rows = {p: i for i, p in enumerate(platforms)}

    colors = {"express": "red", "passenger": "blue", "freight": "orange"}

    fig, ax = plt.subplots(figsize=(14, 10))

    for entry in result.schedule:

        row = rows[entry.platform]

        duration = entry.planned_end - entry.planned_start

        ax.barh(
            row,
            duration,
            left=entry.planned_start,
            height=0.5,
            color=colors.get(train_types[entry.train_id], "gray"),
        )
        ax.text(
            entry.planned_start + duration / 2,
            row,
            entry.train_id,
            ha="center",
            va="center",
            fontsize=8,
        )

    for start, end in snapshot.maintenance_windows:
        ax.axvspan(start, end, alpha=0.2)

    ax.set_yticks(range(len(platforms)))
    ax.set_yticklabels([f"Platform {x}" for x in platforms])
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Platform")
    ax.set_title(
        f"Train Schedule " f"Weighted Delay={result.metrics['weighted_delay']}"
    )
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"[Plot] Saved {output}")


def main():

    print("Creating scenario...")

    snapshot = build_combined_scenario()
    service = SchedulerService({}, snapshot.maintenance_windows)
    result = service.optimize(snapshot, OPTIMIZATION_RUNS)

    print()
    print("BEST SCORE:", result.objective_score)
    print()

    trains = snapshot_train_data(snapshot)
    print_schedule([x.train_id for x in result.schedule], trains)
    plot_schedule(snapshot, result)
    print("Finished.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
