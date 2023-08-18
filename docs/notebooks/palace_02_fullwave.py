# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Full-wave driven simulations with Palace
# Here, we show how Palace may be used to perform full-wave driven simulations in the frequency domain.
# See [Palace – Crosstalk Between Coplanar Waveguides](https://awslabs.github.io/palace/stable/examples/cpw/) for more details.
#
# For a given geometry, one needs to specify the terminals where to apply an excitation similar to Ansys HFSS.
# To this end, lumped ports (or wave ports) have to be added to the geometry to simulate.
# This effectively solves the scattering parameters for the terminals.
#
# ## Installation
# See [Palace – Installation](https://awslabs.github.io/palace/stable/install/) for installation or compilation instructions. Gplugins assumes `palace` is available in your PATH environment variable.
#
# Alternatively, [Singularity / Apptainer](https://apptainer.org/) containers may be used. Instructions for building and an example definition file are found at [Palace – Build using Singularity/Apptainer](https://awslabs.github.io/palace/dev/install/#Build-using-Singularity/Apptainer).
# Afterwards, an easy install method is to add a script to `~/.local/bin` (or elsewhere in `PATH`) calling the Singularity container. For example, one may create a `palace` file containing
# ```console
# #!/bin/bash
# singularity exec ~/palace.sif /opt/palace/bin/palace "$@"
# ```

# %%

import inspect
from collections.abc import Sequence
from math import inf

import gdsfactory as gf
import numpy as np
import pyvista as pv
from gdsfactory.component import Component
from gdsfactory.components.interdigital_capacitor import interdigital_capacitor
from gdsfactory.generic_tech import LAYER, get_generic_pdk
from gdsfactory.technology import LayerStack
from gdsfactory.technology.layer_stack import LayerLevel
from gdsfactory.typings import LayerSpec

from gplugins.palace import run_scattering_simulation_palace

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# LAYER_STACK = PDK.layer_stack

# %% [markdown]
# We employ an example LayerStack used in superconducting circuits similar to {cite:p}`marxer_long-distance_2023`.

# %%
layer_stack = LayerStack(
    layers=dict(
        substrate=LayerLevel(
            layer=LAYER.WAFER,
            thickness=500,
            zmin=0,
            material="Si",
            mesh_order=99,
        ),
        bw=LayerLevel(
            layer=LAYER.WG,
            thickness=200e-3,
            zmin=500,
            material="Nb",
            mesh_order=2,
        ),
    )
)
material_spec = {
    "Si": {"relative_permittivity": 11.45},
    "Nb": {"relative_permittivity": inf},
    "vacuum": {"relative_permittivity": 1},
}

# %%

INTERDIGITAL_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(interdigital_capacitor).parameters.items()
}


@gf.cell
def interdigital_capacitor_enclosed(
    enclosure_box: Sequence[Sequence[float | int]] = [[-200, -200], [200, 200]],
    fingers: int = INTERDIGITAL_DEFAULTS["fingers"],
    finger_length: float | int = INTERDIGITAL_DEFAULTS["finger_length"],
    finger_gap: float | int = INTERDIGITAL_DEFAULTS["finger_gap"],
    thickness: float | int = INTERDIGITAL_DEFAULTS["thickness"],
    cpw_dimensions: Sequence[float | int] = (10, 6),
    gap_to_ground: float | int = 5,
    metal_layer: LayerSpec = INTERDIGITAL_DEFAULTS["layer"],
    gap_layer: LayerSpec = "DEEPTRENCH",
) -> Component:
    """Generates an interdigital capacitor surrounded by a ground plane and coplanar waveguides with ports on both ends.
    See for :func:`~interdigital_capacitor` for details.

    Note:
        ``finger_length=0`` effectively provides a plate capacitor.

    Args:
        enclosure_box: Bounding box dimensions for a ground metal enclosure.
        fingers: total fingers of the capacitor.
        finger_length: length of the probing fingers.
        finger_gap: length of gap between the fingers.
        thickness: Thickness of fingers and section before the fingers.
        gap_to_ground: Size of gap from capacitor to ground metal.
        cpw_dimensions: Dimensions for the trace width and gap width of connecting coplanar waveguides.
        metal_layer: layer for metalization.
        gap_layer: layer for trenching.
    """
    c = Component()
    cap = interdigital_capacitor(
        fingers, finger_length, finger_gap, thickness, metal_layer
    ).ref_center()
    c.add(cap)

    gap = Component()
    for port in cap.get_ports_list():
        port2 = port.copy()
        direction = -1 if port.orientation > 0 else 1
        port2.move((30 * direction, 0))
        port2 = port2.flip()

        cpw_a, cpw_b = cpw_dimensions
        s1 = gf.Section(width=cpw_b, offset=(cpw_a + cpw_b) / 2, layer=gap_layer)
        s2 = gf.Section(width=cpw_b, offset=-(cpw_a + cpw_b) / 2, layer=gap_layer)
        x = gf.CrossSection(
            width=cpw_a,
            offset=0,
            layer=metal_layer,
            port_names=("in", "out"),
            sections=[s1, s2],
        )
        route = gf.routing.get_route(
            port,
            port2,
            cross_section=x,
        )
        c.add(route.references)

        term = c << gf.components.bbox(
            [[0, 0], [cpw_b, cpw_a + 2 * cpw_b]], layer=gap_layer
        )
        if direction < 0:
            term.movex(-cpw_b)
        term.move(
            destination=route.ports[-1].move_copy(-1 * np.array([0, cpw_a / 2 + cpw_b]))
        )

        c.add_port(route.ports[-1])
        c.auto_rename_ports()

    gap.add_polygon(cap.get_polygon_enclosure(), layer=gap_layer)
    gap = gap.offset(gap_to_ground, layer=gap_layer)
    gap = gf.geometry.boolean(A=gap, B=c, operation="A-B", layer=gap_layer)

    ground = gf.components.bbox(bbox=enclosure_box, layer=metal_layer)
    ground = gf.geometry.boolean(
        A=ground, B=[c, gap], operation="A-B", layer=metal_layer
    )

    c << ground

    return c.flatten()


# %%
simulation_box = [[-200, -200], [200, 200]]
c = gf.Component("capacitance_palace")
cap = c << interdigital_capacitor_enclosed(
    metal_layer=LAYER.WG, gap_layer=LAYER.DEEPTRENCH, enclosure_box=simulation_box
)
c.add_ports(cap.ports)
substrate = gf.components.bbox(bbox=simulation_box, layer=LAYER.WAFER)
c << substrate
c.flatten()

# %%
results = run_scattering_simulation_palace(
    c,
    layer_stack=layer_stack,
    driven_settings={
        "MinFreq": 0.1,
        "MaxFreq": 5,
        "FreqStep": 2,
    },
    mesh_parameters=dict(
        background_tag="vacuum",
        background_padding=(0,) * 5 + (700,),
        portnames=c.ports,
        verbosity=1,
        default_characteristic_length=200,
        layer_portname_delimiter=(delimiter := "__"),
        resolutions={
            "bw": {
                "resolution": 14,
            },
            "substrate": {
                "resolution": 50,
            },
            "vacuum": {
                "resolution": 120,
            },
            **{
                f"bw{delimiter}{port}_vacuum": {
                    "resolution": 8,
                }
                for port in c.ports
            },
        },
    ),
)
print(results)

if results.field_file_location:
    field = pv.read(results.field_file_location[0])
    field.slice_orthogonal().plot(scalars="Ue", cmap="turbo")
