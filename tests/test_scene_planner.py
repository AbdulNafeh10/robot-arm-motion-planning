"""Offline smoke test for the CoppeliaSim planning entry point."""

import sys
from types import SimpleNamespace

import numpy as np

from src import plan_scene


class StubSim:
    simulation_stopped = 0
    handle_tree = -2
    handle_all = -1

    def __init__(self):
        self.angles = [0.1] * 6
        self.collision_checks = 0
        self.destroyed = []
        self.next_handle = 30

    def getSimulationState(self):
        return self.simulation_stopped

    def getObject(self, path):
        if path == '/UR5':
            return 1
        if path == '/customizableTable':
            return 2
        return len(path.split('/')) + 10

    def getJointPosition(self, joint):
        return self.angles[joint - 13]

    def setJointPosition(self, joint, angle):
        self.angles[joint - 13] = angle

    def createCollection(self, options):
        handle = self.next_handle
        self.next_handle += 1
        return handle

    def addItemToCollection(self, *args):
        pass

    def checkCollision(self, *args):
        self.collision_checks += 1
        return [0, []]

    def destroyCollection(self, handle):
        self.destroyed.append(handle)


def test_live_entrypoint_restores_joints_and_saves_path(monkeypatch, tmp_path):
    sim = StubSim()
    monkeypatch.setitem(sys.modules, 'coppeliasim_zmqremoteapi_client',
                        SimpleNamespace(RemoteAPIClient=lambda: SimpleNamespace(require=lambda _: sim)))
    monkeypatch.setattr(plan_scene, 'OUTPUT', tmp_path / 'ur5_path.json')

    plan_scene.main()

    np.testing.assert_allclose(sim.angles, [0.1] * 6)
    assert sim.collision_checks > 0
    assert len(sim.destroyed) == 2
    assert plan_scene.OUTPUT.exists()
