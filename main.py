import os
import random
import secrets
from functools import wraps

import ga_scheduler_numba_template as ga
from flask import Flask, jsonify, request, send_from_directory, session


app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SYNAPSE_SECRET_KEY", secrets.token_hex(32))

DEMO_USERS = {
	"admin": {"password": "admin123", "role": "Control room lead"},
	"operator": {"password": "operator123", "role": "Traffic operator"},
}

STATIONS = [
	{"code": "NDLS", "name": "New Delhi", "zone": "Northern Railway"},
	{"code": "CSMT", "name": "Mumbai Central", "zone": "Central Railway"},
	{"code": "HWH", "name": "Howrah Junction", "zone": "Eastern Railway"},
]


def requires_auth(handler):
	@wraps(handler)
	def wrapper(*args, **kwargs):
		if "username" not in session:
			return jsonify({"error": "Authentication required"}), 401
		return handler(*args, **kwargs)

	return wrapper


def serialize_snapshot(snapshot):
	return [
		{
			"train_id": train.train_id,
			"train_type": train.train_type,
			"arrival": train.arrival,
			"delay_minutes": train.delay_minutes,
			"effective_arrival": train.arrival + train.delay_minutes,
			"cross_time": train.cross_time,
			"platform": train.platform,
			"priority": train.priority or ga.PRIORITY_WEIGHTS[train.train_type],
		}
		for train in snapshot.trains
	]


def build_station_snapshot(station_code):
	station = next((item for item in STATIONS if item["code"] == station_code), None)
	if station is None:
		raise ValueError("Unknown station")

	rng = random.Random(ga.RANDOM_SEED + sum(ord(char) for char in station_code))
	train_types = ("express", "passenger", "passenger", "freight")
	trains = tuple(
		ga.Train(
			train_id=f"{station_code}-{index}",
			train_type=rng.choice(train_types),
			arrival=rng.randint(600, 660),
			cross_time=rng.randint(2, 6),
			platform=rng.randint(1, 12),
			delay_minutes=rng.randint(ga.MIN_CURRENT_DELAY, ga.MAX_CURRENT_DELAY),
		)
		for index in range(1, ga.TRAIN_COUNT + 1)
	)
	maintenance_start = rng.randint(620, 640)
	return ga.SchedulerSnapshot(
		snapshot_id=f"{station_code.lower()}-demo",
		trains=trains,
		minimum_headway_minutes=ga.MIN_HEADWAY,
		maintenance_windows=((maintenance_start, maintenance_start + 5),),
	)


def run_optimizer(station_code):
	snapshot = build_station_snapshot(station_code)
	numba_data = ga.prepare_numba_data(ga.snapshot_train_data(snapshot))
	chromosome, score, generations = ga.run_ga_fast(
		numba_data,
		generations=60,
		pop_size=180,
		mutation_rate=ga.MUTATION_RATE,
		stagnation_limit=ga.STAGNATION_LIMIT,
		minimum_headway=snapshot.minimum_headway_minutes,
	)
	train_ids = numba_data[0]
	schedule = [train_ids[index] for index in chromosome]
	entries = ga.build_schedule_entries(schedule, snapshot)
	current_delay = sum(entry.current_delay_minutes for entry in entries)
	new_delay = sum(entry.delay_minutes for entry in entries)
	weighted_delay = sum(entry.weighted_delay for entry in entries)
	return {
		"snapshot_id": snapshot.snapshot_id,
		"station": next(item for item in STATIONS if item["code"] == station_code),
		"trains": serialize_snapshot(snapshot),
		"maintenance_windows": [list(window) for window in snapshot.maintenance_windows],
		"schedule": [
			{
				"train_id": entry.train_id,
				"sequence": entry.sequence,
				"platform": entry.platform,
				"planned_start": entry.planned_start,
				"planned_end": entry.planned_end,
				"delay_minutes": entry.delay_minutes,
				"weighted_delay": entry.weighted_delay,
				"current_delay_minutes": entry.current_delay_minutes,
				"total_delay_minutes": entry.total_delay_minutes,
			}
			for entry in entries
		],
		"metrics": {
			"weighted_delay": weighted_delay,
			"current_delay": current_delay,
			"new_delay": new_delay,
			"total_delay": current_delay + new_delay,
			"average_delay": (current_delay + new_delay) / len(entries),
			"trains_scheduled": len(entries),
			"generations_completed": generations,
			"algorithm": "Genetic algorithm",
			"objective_score": float(score),
		},
	}


@app.get("/")
def index():
	return send_from_directory(app.static_folder, "index.html")


@app.post("/api/login")
def login():
	credentials = request.get_json(silent=True) or {}
	username = credentials.get("username", "").strip().lower()
	password = credentials.get("password", "")
	user = DEMO_USERS.get(username)
	if user is None or not secrets.compare_digest(password, user["password"]):
		return jsonify({"error": "Invalid username or password"}), 401
	session["username"] = username
	return jsonify({"username": username, "role": user["role"]})


@app.post("/api/logout")
def logout():
	session.clear()
	return jsonify({"ok": True})


@app.get("/api/session")
def current_session():
	username = session.get("username")
	if username is None:
		return jsonify({"authenticated": False})
	return jsonify({
		"authenticated": True,
		"username": username,
		"role": DEMO_USERS[username]["role"],
	})


@app.get("/api/dashboard")
@requires_auth
def dashboard():
	station_code = request.args.get("station", "").upper()
	if not station_code:
		return jsonify({"error": "Select a station first"}), 400
	try:
		return jsonify(run_optimizer(station_code))
	except ValueError as error:
		return jsonify({"error": str(error)}), 404


@app.get("/api/stations")
def stations():
	return jsonify(STATIONS)


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=5000, debug=True)
