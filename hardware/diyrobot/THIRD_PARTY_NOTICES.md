# DIYRobot Third-Party Notices

This document distinguishes mechanical provenance from software dependencies.
It does not replace the license text shipped with any included third-party
component.

## TRLC-DK1 Mechanical Basis

The DIYRobot follower-arm mechanism is a modified design derived from the
TRLC-DK1 Follower by The Robot Learning Company.

- Upstream repository: <https://github.com/robot-learning-co/trlc-dk1>
- Audited upstream revision: `12f5368aefd0381461f2c7ffbb5611b4e8c90de9`
- Upstream license: Apache License 2.0
- Upstream copyright: Copyright 2025-2026 The Robot Learning Company UG
  (haftungsbeschrankt). All rights reserved.

The local working assembly identifies `TRLC-DK1-Follower_v0.2.0` as its
starting version. The current upstream repository publishes v0.3.0; those two
geometries must not be treated as identical. The exact historical upstream
commit used to create the local v0.2.0 working copy has not yet been recorded.

The DIYRobot adaptation changes actuator interfaces and associated arm parts
for RobStride O3 integration. Every released modified CAD file must retain the
upstream attribution and carry a prominent modification notice, as required by
Apache-2.0 section 4.

## Original DIYRobot Mechanics

The three-omni-wheel mobile chassis and lift-system mechanical integration are
original DIYRobot engineering. This statement applies to those mechanical
assemblies only. It does not claim ownership of purchased actuators, cameras,
interfaces, vendor protocols, or upstream software.

## Software and Driver Dependencies

The lower-host release integrates or interoperates with:

- LeRobot, Apache License 2.0.
- Damiao motor-control code and protocols; the TRLC-DK1 upstream notice also
  identifies DM_Control_Python under the MIT License.
- RobStride actuator transports and vendor interfaces, subject to their
  respective upstream or vendor terms.
- Feetech servo interfaces, subject to their respective upstream or vendor
  terms.

The repository-level license files remain authoritative for included software.
Manufacturer names and model identifiers are used only to describe hardware
compatibility and do not imply endorsement.

## CAD Redistribution Boundary

Do not publish vendor-native motor, bearing, camera, rail, or fastener CAD merely
because it appears inside an internal assembly. When redistribution permission
is unclear, publish the manufacturer, exact part number, interface dimensions,
and procurement link in the BOM, and require the reproducer to obtain the model
from its owner.
