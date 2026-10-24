import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from paz.models.foundation.kimodo import mujoco as export
from paz.models.foundation.kimodo import skeleton

XML_PATH = Path(__file__).parents[4] / "examples/rl_gym/assets/g1_29dof.xml"


def read_xml_hinges():
    root = ET.parse(XML_PATH).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    classes = {}
    for default in root.iter("default"):
        joint = default.find("joint")
        if default.get("class") and joint is not None:
            classes[default.get("class")] = joint.get("axis")
    hinges = []
    for joint in root.find("worldbody").iter("joint"):
        if joint.get("type") == "free":
            continue
        axis = joint.get("axis") or classes[joint.get("class")]
        body_quaternion = parents[joint].get("quat")
        hinges.append((joint.get("name"), axis, body_quaternion))
    return hinges


def test_hinge_tables_match_the_unitree_xml():
    hinges = read_xml_hinges()
    assert len(hinges) == len(export.HINGE_JOINTS) == 29
    for (name, axis, body_quat), hinge, axis_name in zip(
            hinges, export.HINGE_JOINTS, export.HINGE_AXES):
        assert name == f"{hinge}_joint"
        expected_axis = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[axis_name]
        assert axis == expected_axis
        literal = export.BODY_QUATERNIONS.get(hinge)
        if body_quat is None:
            assert literal is None
        else:
            parsed = [float(value) for value in body_quat.split()]
            assert np.allclose(parsed, literal)


def test_rotation_offsets_are_rotations():
    determinants = jp.linalg.det(export.ROTATION_OFFSETS)
    assert jp.allclose(determinants, 1.0, atol=1e-5)
    assert jp.allclose(jp.linalg.norm(export.HINGE_AXIS_VECTORS, axis=-1), 1.0)


def test_identity_pose_maps_to_mujoco_frame():
    identities = jp.tile(jp.eye(3), (skeleton.NUM_JOINTS, 1, 1))
    qpos = export.to_qpos(identities, jp.array([0.1, 0.8, 0.3]))
    assert qpos.shape == (36,)
    assert jp.allclose(qpos[:3], jp.array([0.3, 0.1, 0.8]))
    assert jp.allclose(qpos[3:7], jp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    plain = [h for h, j in enumerate(export.HINGE_JOINTS)
             if j not in export.BODY_QUATERNIONS]
    assert jp.allclose(qpos[7:][jp.array(plain)], 0.0, atol=1e-6)


def test_root_rotation_about_kimodo_up_is_a_yaw_in_mujoco():
    from paz import SO3
    rotations = jp.tile(jp.eye(3), (skeleton.NUM_JOINTS, 1, 1))
    rotations = rotations.at[0].set(SO3.rotation_y(0.5))
    qpos = export.to_qpos(rotations, jp.zeros(3))
    expected = jp.array([jp.cos(0.25), 0.0, 0.0, jp.sin(0.25)])
    assert jp.allclose(qpos[3:7], expected, atol=1e-6)


def test_to_qpos_vmaps_over_frames():
    rotations = jp.tile(jp.eye(3), (4, skeleton.NUM_JOINTS, 1, 1))
    qpos = jax.vmap(export.to_qpos)(rotations, jp.zeros((4, 3)))
    assert qpos.shape == (4, 36)
