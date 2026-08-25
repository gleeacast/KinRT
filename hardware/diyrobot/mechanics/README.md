# DIYRobot Mechanical Release

## Provenance Boundary

The DIYRobot hardware contains two mechanically distinct contributions:

1. The follower-arm mechanism is derived from the Apache-2.0-licensed
   [TRLC-DK1](https://github.com/robot-learning-co/trlc-dk1) Follower design and
   adapted for RobStride O3 actuators.
2. The three-omni-wheel mobile base and lift integration are original DIYRobot
   mechanical designs.

This distinction must remain visible in the repository, generated drawings,
CAD metadata, release notes, and website.

## Current CAD Audit

The private mechanical workspace contains 155 Siemens NX `.prt` files, one
STEP assembly, and one NX import log. It is not copied into this public folder
because it is a working directory rather than an approved release package.

- The STEP assembly is named `TRLC-DK1-Follower_v0.2.0.step` and still contains
  Damiao DM-J4340/DM-J4310 component identities.
- Multiple NX parts were modified after that STEP was exported, including arm
  links, motor adapters, shaft and bearing extensions, cable covers, and mounts.
- NX-generated `577xxx` duplicates represent import history and are not a clean
  public source layout.
- The NX log exposes workstation paths and must never be distributed.
- No approved mobile-base CAD was present in the audited mechanical folder.

Consequently, the old STEP is not presented as the final RobStride-adapted arm,
and no unverified chassis dimensions are inferred from the control code or
photographs.

## Upstream References

- [TRLC-DK1 repository](https://github.com/robot-learning-co/trlc-dk1)
- [Current upstream Follower STEP](https://github.com/robot-learning-co/trlc-dk1/blob/main/hardware/TRLC-DK1-Follower_v0.3.0.step)
- [Current upstream Follower URDF](https://github.com/robot-learning-co/trlc-dk1/tree/main/urdf/follower)
- [TRLC-DK1 Apache-2.0 license](https://github.com/robot-learning-co/trlc-dk1/blob/main/LICENSE)

The upstream v0.3.0 files are references. They are not a substitute for the
DIYRobot v0.2.0-derived final assembly.

## Release Acceptance Checklist

Before placing CAD in this directory:

1. Open the clean NX assembly candidate and resolve every component reference.
2. Compare the assembly against both physical follower arms, including mirrored
   parts, motor orientation, cable routes, wrist camera mounts, and grippers.
3. Confirm RobStride O3 interface dimensions and remove unapproved vendor CAD.
4. Export a new neutral STEP from the verified assembly.
5. Export one oriented STL and one 3MF per printable part.
6. Generate an exploded assembly drawing and fastener callouts.
7. Record material, layer height, wall count, infill, support, tolerance, insert,
   and post-processing requirements.
8. Add the original chassis and lift CAD, drawings, and manufacturing notes.
9. Build one release-candidate platform using only the public package.
10. Preserve TRLC-DK1 attribution and mark every modified derivative file.

## Intended Public Layout

```text
mechanics/
├── README.md
├── arm/
│   ├── source/
│   ├── step/
│   ├── stl/
│   ├── 3mf/
│   └── drawings/
├── base/
│   ├── source/
│   ├── step/
│   └── drawings/
└── lift/
    ├── source/
    ├── step/
    └── drawings/
```

See [the public parameter register](../PARAMETERS.md), [the core BOM](../BOM.csv),
and [third-party notices](../THIRD_PARTY_NOTICES.md) for the values that can be
released before final CAD approval.
