"""Plan a UR5 tool-reference approach to the surface of the Goal sphere.

Use with the custom UR5 scene and the simulation STOPPED.
Commands: python -m src.touch_goal check | plan | play
This is kinematic preview, not a physical contact/force-control system.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .collision import edge_is_free
from .rrt import plan_rrt_connect
from .play_scene import animate_segment

OUTPUT = Path(__file__).resolve().parents[1] / 'output' / 'touch_goal_path.json'
OBSTACLES = ('RobotTable', 'GoalTable', 'GoalStand', 'diningChair', 'laptop', 'projector')
GOAL = 'Goal'
TOOL = '/UR5/connection'
POSITION_TOLERANCE = 0.008


def sphere_surface(center, radius, from_position):
    """Point on sphere nearest an approach position."""
    center = np.asarray(center, dtype=float)
    direction = np.asarray(from_position, dtype=float) - center
    distance = float(np.linalg.norm(direction))
    if radius <= 0 or distance < 1e-9:
        raise ValueError('Sphere radius and approach direction must be valid')
    return center + radius * direction / distance


def sphere_radius(sim, handle):
    spans = [sim.getObjectFloatParam(handle, high) -
             sim.getObjectFloatParam(handle, low)
             for low, high in (
                 (sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_max_x),
                 (sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_max_y),
                 (sim.objfloatparam_objbbox_min_z, sim.objfloatparam_objbbox_max_z))]
    if min(spans) <= 0 or max(spans) - min(spans) > 0.005:
        raise RuntimeError('Goal does not have a roughly spherical bounding box')
    return min(spans) / 2


class Scene:
    def __init__(self, sim):
        self.sim = sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('Stop simulation before using this script')
        self.robot = sim.getObject('/UR5')
        self.tool = sim.getObject(TOOL)
        self.goal = sim.getObject('/' + GOAL)
        self.obstacles = {name: sim.getObject('/' + name) for name in OBSTACLES}
        self.joints = [sim.getObject('/UR5/' + '/'.join(['joint'] * i))
                       for i in range(1, 7)]
        self.original = np.array([sim.getJointPosition(j) for j in self.joints])
        self.goal_properties = sim.getObjectSpecialProperty(self.goal)
        self.robot_collection = None
        try:
            # The sphere represents the intentional contact target, not an obstacle.
            sim.setObjectSpecialProperty(
                self.goal, self.goal_properties & ~sim.objectspecialproperty_collidable)
            self.robot_collection = sim.createCollection(0)
            sim.addItemToCollection(self.robot_collection, sim.handle_tree, self.robot, 0)
        except Exception:
            self.restore()
            raise
        self.center = np.asarray(sim.getObjectPosition(self.goal, sim.handle_world), float)
        self.radius = sphere_radius(sim, self.goal)
        self.start_xyz = self.xyz()
        self.surface = sphere_surface(self.center, self.radius, self.start_xyz)
        self.start = np.rad2deg(self.original)
        self.checks = 0

    def pose(self, deg):
        for joint, angle in zip(self.joints, np.deg2rad(deg)):
            self.sim.setJointPosition(joint, float(angle))

    def xyz(self):
        return np.asarray(self.sim.getObjectPosition(self.tool, self.sim.handle_world), float)

    def free(self, deg):
        self.pose(deg)
        result = self.sim.checkCollision(self.robot_collection, self.sim.handle_all)[0]
        self.checks += 1
        if result not in (0, 1):
            raise RuntimeError('Invalid simulator collision result: ' + str(result))
        return result == 0

    def fingerprint(self):
        names = {**self.obstacles, GOAL: self.goal}
        return {name: {'position': np.round(self.sim.getObjectPosition(h, -1), 5).tolist(),
                       'orientation': np.round(self.sim.getObjectOrientation(h, -1), 5).tolist()}
                for name, h in names.items()}

    def restore(self):
        for joint, angle in zip(getattr(self, 'joints', []), getattr(self, 'original', [])):
            self.sim.setJointPosition(joint, float(angle))
        if getattr(self, 'robot_collection', None) is not None:
            self.sim.destroyCollection(self.robot_collection)
            self.robot_collection = None
        if hasattr(self, 'goal_properties'):
            self.sim.setObjectSpecialProperty(self.goal, self.goal_properties)


def solve_goal(scene, count=10):
    sim = scene.sim
    initial = scene.original
    seeds = [initial, np.deg2rad([0, -40, -45, -15, 90, 175]),
             np.deg2rad([35, -55, 60, 0, 90, 90]),
             np.deg2rad([-35, -55, 60, 0, 90, 90])]
    rng = np.random.default_rng(42)
    for _ in range(count-len(seeds)):
        seeds.append(np.clip(initial + rng.normal(0, 0.65, 6), -3.05, 3.05))
    candidates = []
    for i, seed in enumerate(seeds, 1):
        seed = np.clip(seed, -3.1, 3.1)
        def residual(q):
            scene.pose(np.rad2deg(q))
            return np.r_[scene.xyz() - scene.surface, 0.002 * (q-seed)]
        fit = least_squares(residual, seed, bounds=(-np.pi*np.ones(6), np.pi*np.ones(6)),
                            max_nfev=120, diff_step=1e-4)
        deg = np.rad2deg(fit.x)
        scene.pose(deg)
        error = float(np.linalg.norm(scene.xyz() - scene.surface))
        clear = scene.free(deg)
        result = {'angles_deg': np.round(deg, 5).tolist(),
                  'error_m': round(error, 6), 'collision_free': clear}
        candidates.append(result)
        print(f'IK {i}/{len(seeds)}: error {error*100:.2f} cm, clear: {clear}', flush=True)
    accepted = [c for c in candidates if c['collision_free'] and
                c['error_m'] <= POSITION_TOLERANCE]
    accepted.sort(key=lambda c: (c['error_m'],
                   np.linalg.norm(np.asarray(c['angles_deg']) - scene.start)))
    return (accepted[0] if accepted else None), candidates


def verify_scene(scene, saved):
    if scene.fingerprint() != saved['objects']:
        raise RuntimeError('Object positions/orientations have changed; replan')
    if not np.allclose(scene.start, saved['start_deg'], atol=0.1):
        raise RuntimeError('Robot does not match saved start pose; replan')
    if np.linalg.norm(scene.surface - saved['touch_point_m']) > 0.002:
        raise RuntimeError('Goal surface changed; replan')


def main(action):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient().require('sim')
    scene = Scene(sim)
    try:
        print('Tool:', np.round(scene.start_xyz, 3).tolist())
        print('Sphere center:', np.round(scene.center, 3).tolist(),
              'radius:', round(scene.radius, 4), 'm')
        print('Contact reference:', np.round(scene.surface, 3).tolist())
        print('Checking all non-target scene collidables...', flush=True)
        if not scene.free(scene.start):
            raise RuntimeError('Current robot start configuration is in collision. No planning.')
        if action in ('check', 'plan'):
            best, candidates = solve_goal(scene)
            if best is None:
                print('No accurate collision-free goal pose. No playback. Adjust scene or IK seeds.')
                return
            print('Verified goal position:', best['angles_deg'], 'error:', best['error_m'], 'm')
            if action == 'check':
                print('Endpoint check only. No route generated or played.')
                return
            start = scene.start.copy()
            viable = [c for c in candidates if c['collision_free'] and
                      c['error_m'] <= POSITION_TOLERANCE]
            viable.sort(key=lambda c: np.linalg.norm(np.asarray(c['angles_deg'])-start))
            # Try a few different valid goal arm configurations; not only the
            # one with the smallest position error.
            goals = viable[:2]
            path = None
            direct = False
            selected_goal = None
            for goal_number, option in enumerate(goals, 1):
                goal = np.asarray(option['angles_deg'])
                print(f'Testing goal configuration {goal_number}/{len(goals)}...', flush=True)
                direct = edge_is_free(start, goal, scene.free, resolution=1.0)
                print('Direct joint path:', 'CLEAR' if direct else 'BLOCKED', flush=True)
                if direct:
                    print('Skipping direct route for obstacle-avoidance demo.', flush=True)
                    continue
                for seed in (9, 19):
                    print(f'Bidirectional RRT, seed {seed}, max 15 seconds...', flush=True)
                    path = plan_rrt_connect(
                        start, goal, scene.free, limits=(-180, 180),
                        step_size=18, edge_resolution=2,
                        max_iterations=400, max_seconds=15, seed=seed,
                        progress=lambda iteration, a, b, elapsed: print(
                            f'  iteration {iteration}, trees {a}/{b} nodes, {elapsed:.0f}s',
                            flush=True))
                    if path is not None:
                        selected_goal = goal
                        break
                if path is not None:
                    break
            if path is None:
                print('No collision-free route found within the search limits.')
                print('Playback disabled. This does not prove the goal is unreachable.')
                return
            goal = selected_goal
            for a, b in zip(path[:-1], path[1:]):
                if not edge_is_free(a, b, scene.free, resolution=0.5):
                    raise RuntimeError('Fine path validation failed. No saved route.')
            data = {'objects': scene.fingerprint(), 'start_deg': start.tolist(),
                    'goal_deg': goal.tolist(), 'sphere_center_m': scene.center.tolist(),
                    'sphere_radius_m': scene.radius, 'touch_point_m': scene.surface.tolist(),
                    'waypoints_deg': np.round(path, 5).tolist(),
                    'collision_resolution_deg': 0.5, 'direct_path_clear': direct}
            OUTPUT.parent.mkdir(exist_ok=True)
            OUTPUT.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
            print(f'Saved validated candidate with {len(path)} waypoints: {OUTPUT}')
            print('Next: py -m src.touch_goal play (only after reviewing this output).')
            return
        if not OUTPUT.exists():
            raise RuntimeError('No route. Run python -m src.touch_goal plan first')
        saved = json.loads(OUTPUT.read_text(encoding='utf-8'))
        verify_scene(scene, saved)
        path = np.asarray(saved['waypoints_deg'], float)
        if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
            raise RuntimeError('Invalid saved path')
        if not np.allclose(path[0], saved['start_deg'], atol=0.01) or not np.allclose(
                path[-1], saved['goal_deg'], atol=0.01):
            raise RuntimeError('Route endpoints do not match saved plan')
        for a, b in zip(path[:-1], path[1:]):
            if not edge_is_free(a, b, scene.free, resolution=0.5):
                raise RuntimeError('Preflight collision detected. Playback cancelled.')
        print('Preflight clear. Kinematic movement begins now.', flush=True)
        scene.pose(path[0])
        for i, (a, b) in enumerate(zip(path[:-1], path[1:]), 1):
            duration = max(2, np.max(np.abs(b-a)) / 12)
            animate_segment(a, b, scene.pose, seconds=duration, fps=20)
            print(f'Segment {i}/{len(path)-1} complete', flush=True)
        print('At contact reference. Holding for 3 seconds.', flush=True)
        time.sleep(3)
        for a, b in zip(path[:0:-1], path[-2::-1]):
            duration = max(2, np.max(np.abs(b-a)) / 12)
            animate_segment(a, b, scene.pose, seconds=duration, fps=20)
        print('Returned to starting pose.')
    finally:
        scene.restore()
        print('Original robot pose and Goal settings restored.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'plan', 'play'))
    main(parser.parse_args().action)
