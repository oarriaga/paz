# G1 asset provenance

`g1_29dof.xml` was derived from
`bulkington/mobi/robots/models/unitree_g1/g1_29dof.xml` at commit
`ac7c5a77aa7f95ca95f2c073cb72fec36ebef943`.

Source SHA-256:
`2fd0934d70374f095a3d625c7680a9d7ae938f69aad7e165b4afefc038152272`.

Bundled SHA-256:
`0a0061818bff68d4a4f7443ece3307dd10b980a663cdbf483c93e127a8bd3d91`.

The bundled file removes external mesh assets and non-colliding visual mesh
geometries. It retains the full body and joint tree, joint limits,
actuators, sensors, and all named collision primitives. Three inertials
(`torso_link`, `waist_yaw_link`, `waist_roll_link`) are replaced by the
values of `g1_29dof_rev_1_0.urdf`, the revision the reference policy was
trained on: the source torso weighs 9.598 kg against 7.817 kg with the
head composited in the URDF. A floor,
lighting, and simple materials make the file standalone. The source license is
preserved in `UNITREE_LICENSE`.
