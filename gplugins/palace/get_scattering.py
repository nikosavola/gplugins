from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
from collections.abc import Collection, Mapping, Sequence
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import gdsfactory as gf
import gmsh
import pandas as pd
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.technology import LayerStack
from numpy import isfinite

from gplugins.typings import DrivenFullWaveResults, RFMaterialSpec

DRIVE_JSON = "driven.json"
DRIVEN_TEMPLATE = Path(__file__).parent / DRIVE_JSON


def _generate_json(
    simulation_folder: Path,
    name: str,
    bodies: dict[str, dict[str, Any]],
    absorbing_surfaces: Collection[str],
    layer_stack: LayerStack,
    material_spec: RFMaterialSpec,
    element_order: int,
    physical_name_to_dimtag_map: dict[str, tuple[int, int]],
    metal_surfaces: Collection[str],
    background_tag: str | None = None,
    edge_signals: Sequence[Sequence[str]] | None = None,
    internal_signals: Sequence[Sequence[str]] | None = None,
    simulator_params: Mapping[str, Any] | None = None,
    driven_settings: Mapping[str, float | int | bool] | None = None,
    adaptive_mesh_iterations: int | None = None,
) -> list[Path]:
    """Generates a json file for full-wave Palace simulations."""
    # TODO: Generalise to merger with the Elmer implementations"""
    used_materials = {v.material for v in layer_stack.layers.values()} | (
        {background_tag} if background_tag else {}
    )
    used_materials = {
        k: material_spec[k]
        for k in used_materials
        if isfinite(material_spec[k].get("relative_permittivity", inf))
    }

    with open(DRIVEN_TEMPLATE) as fp:
        palace_json_data = json.load(fp)

    material_to_attributes_map = {
        v["material"]: physical_name_to_dimtag_map[k][1] for k, v in bodies.items()
    }

    palace_json_data["Model"]["Mesh"] = f"{name}.msh"
    if adaptive_mesh_iterations:
        palace_json_data["Model"]["Refinement"] = {}
        palace_json_data["Model"]["Refinement"][
            "UniformLevels"
        ] = adaptive_mesh_iterations
    palace_json_data["Domains"]["Materials"] = [
        {
            "Attributes": [material_to_attributes_map.get(material, None)],
            "Permittivity": props["relative_permittivity"],
            "Permeability": props["relative_permeability"],
            "LossTan": props.get("loss_tangent", 0.0),
            "Conductivity": props.get("conductivity", 0.0),
        }
        for material, props in used_materials.items()
    ]
    # TODO list here attributes that contained LossTAN
    # palace_json_data["Domains"]["Postprocessing"]["Dielectric"] = [
    # ]

    palace_json_data["Boundaries"]["PEC"] = {
        "Attributes": [
            physical_name_to_dimtag_map[layer][1]
            for layer in set(metal_surfaces)
            - set(itertools.chain.from_iterable(edge_signals or []))
            - set(itertools.chain.from_iterable(internal_signals or []))
            - set(absorbing_surfaces or [])
        ]
    }

    # Farfield surface
    palace_json_data["Boundaries"]["Absorbing"] = {
        "Attributes": [
            physical_name_to_dimtag_map[e][1] for e in absorbing_surfaces
        ],  # TODO get farfield _None etc
        "Order": 1,
    }
    # TODO palace_json_data["Boundaries"]["Postprocessing"]["Dielectric"]

    palace_json_data["Solver"]["Order"] = element_order
    if driven_settings is not None:
        palace_json_data["Solver"]["Driven"] |= driven_settings
    if simulator_params is not None:
        palace_json_data["Solver"]["Linear"] |= simulator_params

    # need one simulation per port to excite, see https://github.com/awslabs/palace/issues/81
    jsons = []
    for port in itertools.chain(edge_signals or [], internal_signals or []):
        port_i = 0
        if edge_signals:
            palace_json_data["Boundaries"]["WavePort"] = [
                {
                    "Index": (port_i := port_i + 1),
                    "Attributes": [
                        physical_name_to_dimtag_map[signal][1]
                        for signal in signal_group
                    ],
                    "Excitation": port == signal_group,
                    "Mode": 1,
                    "Offset": 0.0,
                }
                for signal_group in edge_signals
            ]
        if internal_signals:
            palace_json_data["Boundaries"]["LumpedPort"] = [
                {
                    "Index": (port_i := port_i + 1),
                    "Attributes": [
                        physical_name_to_dimtag_map[signal][1]
                        for signal in signal_group
                    ],
                    "Excitation": port == signal_group,
                    "Direction": "+X",  # TODO infer from ground to trace direction?
                    "R": 50,
                }
                for signal_group in internal_signals
            ]
        # TODO regex here is hardly robust
        port_name = re.search(r"__(.*?)___", port[0]).group(1)
        palace_json_data["Problem"]["Output"] = f"postpro_{port_name}"

        with open(
            (json_name := simulation_folder / f"{name}_{port_name}.json"),
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(palace_json_data, fp, indent=4)
        jsons.append(json_name)

    return jsons


def _palace(
    simulation_folder: Path, json_files: Collection[Path | str], n_processes: int = 1
):
    """Run simulations with Palace."""

    # split processes as evenly as possible
    quotient, remainder = divmod(n_processes, len(json_files))
    n_processes_per_json = [quotient] * len(json_files)
    for i in range(remainder):
        n_processes_per_json[i] = max(
            n_processes_per_json[i] + 1, 1
        )  # need at least one

    palace = shutil.which("palace")
    if palace is None:
        raise RuntimeError("palace not found. Make sure it is available in your PATH.")
    # TODO handle better than this. Ideally distributed and scheduled with @ray.remote
    for json_file, n_processes_json in zip(json_files, n_processes_per_json):
        with open(str(json_file) + "_palace.log", "w", encoding="utf-8") as fp:
            # TODO raise error on actual failure
            p = subprocess.Popen(
                [palace, str(json_file)]
                if n_processes == 1
                else [palace, "-np", str(n_processes_json), str(json_file)],
                cwd=simulation_folder,
                stdout=fp,
                stderr=fp,
            )
    p.communicate()  # wait only for the last iteration


def _read_palace_results(
    simulation_folder: Path,
    mesh_filename: str,
    ports: Collection[str],
    is_temporary: bool,
) -> DrivenFullWaveResults:
    """Fetch results from successful Palace simulations."""
    # TODO combine different matrices
    scattering_matrix = pd.DataFrame()
    for port in ports:
        scattering_matrix = pd.concat(
            [
                scattering_matrix,
                pd.read_csv(
                    simulation_folder / f"postpro_{port}" / "port-S.csv", dtype=float
                ),
            ],
            axis="columns",
        )
    scattering_matrix = (
        scattering_matrix.T.drop_duplicates().T
    )  # Remove duplicate freqs.
    DrivenFullWaveResults.update_forward_refs()
    return DrivenFullWaveResults(
        scattering_matrix=scattering_matrix,  # TODO maybe convert to SDict or similar from DataFrame
        **(
            {}
            if is_temporary
            else dict(
                mesh_location=simulation_folder / mesh_filename,
                field_file_location=[
                    simulation_folder
                    / f"postpro_{port}"
                    / "paraview"
                    / "driven"
                    / "driven.pvd"
                    for port in ports
                ],
            )
        ),
    )


def run_scattering_simulation_palace(
    component: gf.Component,
    element_order: int = 1,
    n_processes: int = 1,
    layer_stack: LayerStack | None = None,
    material_spec: RFMaterialSpec | None = None,
    simulation_folder: Path | str | None = None,
    simulator_params: Mapping[str, Any] | None = None,
    driven_settings: Mapping[str, float | int | bool] | None = None,
    adaptive_mesh_iterations: int | None = None,
    mesh_parameters: dict[str, Any] | None = None,
    mesh_file: Path | str | None = None,
) -> DrivenFullWaveResults:
    """Run full-wave finite element method simulations using
    `Palace`_.
    Returns the field solution and resulting scattering matrix.

    .. note:: You should have `palace` in your PATH.

    Args:
        component: Simulation environment as a gdsfactory component.
        element_order:
            Order of polynomial basis functions.
            Higher is more accurate but takes more memory and time to run.
        n_processes: Number of processes to use for parallelization
        layer_stack:
            :class:`~LayerStack` defining defining what layers to include in the simulation
            and the material properties and thicknesses.
        material_spec:
            :class:`~RFMaterialSpec` defining material parameters for the ones used in ``layer_stack``.
        simulation_folder:
            Directory for storing the simulation results.
            Default is a temporary directory.
        simulator_params: Palace-specific parameters. This will be expanded to ``solver["Linear"]`` in
            the Palace config, see `Palace documentation <https://awslabs.github.io/palace/stable/config/solver/#solver[%22Linear%22]>`_
        driven_settings: Driven full-wave parameters in Palace. This will be expanded to ``solver["Driven"]`` in
            the Palace config, see `Palace documentation <https://awslabs.github.io/palace/stable/config/solver/#solver[%22Driven%22]>`_
        adaptive_mesh_iterations: Iterations to use for adaptive meshing.
        mesh_parameters:
            Keyword arguments to provide to :func:`~Component.to_gmsh`.
        mesh_file: Path to a ready mesh to use. Useful for reusing one mesh file.
            By default a mesh is generated according to ``mesh_parameters``.

    .. _Palace https://github.com/awslabs/palace
    """

    if layer_stack is None:
        layer_stack = LayerStack(
            layers={
                k: LAYER_STACK.layers[k]
                for k in (
                    "core",
                    "substrate",
                    "box",
                )
            }
        )
    if material_spec is None:
        material_spec: RFMaterialSpec = {
            "si": {"relative_permittivity": 11.45},
            "sio2": {"relative_permittivity": 1},
            "vacuum": {"relative_permittivity": 1},
        }

    temp_dir = TemporaryDirectory()
    simulation_folder = Path(simulation_folder or temp_dir.name)
    simulation_folder.mkdir(exist_ok=True, parents=True)

    filename = component.name + ".msh"
    if mesh_file:
        shutil.copyfile(str(mesh_file), str(simulation_folder / filename))
    else:
        component.to_gmsh(
            type="3D",
            filename=simulation_folder / filename,
            layer_stack=layer_stack,
            gmsh_version=2.2,  # see https://mfem.org/mesh-formats/#gmsh-mesh-formats
            **(mesh_parameters or {}),
        )

    # re-read the mesh
    gmsh.initialize(interruptible=False)
    gmsh.merge(str(simulation_folder / filename))
    mesh_surface_entities = {
        gmsh.model.getPhysicalName(*dimtag)
        for dimtag in gmsh.model.getPhysicalGroups(dim=2)
    }

    # Signals are converted to Boundaries
    ground_layers = {
        next(k for k, v in layer_stack.layers.items() if v.layer == port.layer)
        for port in component.get_ports()
    }  # ports allowed only on metal
    # TODO infer port delimiter from somewhere
    port_delimiter = "__"
    metal_surfaces = [
        e for e in mesh_surface_entities if any(ground in e for ground in ground_layers)
    ]
    # Group signal BCs by ports
    metal_signal_surfaces_grouped = [
        [e for e in metal_surfaces if port in e] for port in component.ports
    ]
    metal_ground_surfaces = set(metal_surfaces) - set(
        itertools.chain.from_iterable(metal_signal_surfaces_grouped)
    )

    ground_layers |= metal_ground_surfaces

    # dielectrics
    bodies = {
        k: {
            "material": v.material,
        }
        for k, v in layer_stack.layers.items()
        if port_delimiter not in k and k not in ground_layers
    }
    if background_tag := (mesh_parameters or {}).get("background_tag", "vacuum"):
        bodies = {**bodies, background_tag: {"material": background_tag}}

    # TODO refactor to not require this map, the same information could be transferred with the variables above
    physical_name_to_dimtag_map = {
        gmsh.model.getPhysicalName(*dimtag): dimtag
        for dimtag in gmsh.model.getPhysicalGroups()
    }
    absorbing_surfaces = {
        k for k in physical_name_to_dimtag_map.keys() if "___None" in k
    } - set(
        ground_layers
    )  # keep metal edge as PEC

    gmsh.finalize()

    jsons = _generate_json(
        simulation_folder,
        component.name,
        bodies,
        absorbing_surfaces,
        layer_stack,
        material_spec,
        element_order,
        physical_name_to_dimtag_map,
        metal_surfaces,
        background_tag,
        None,  # TODO edge
        metal_signal_surfaces_grouped,  # internal
        simulator_params,
        driven_settings,
        adaptive_mesh_iterations,
    )
    _palace(simulation_folder, jsons, n_processes)
    results = _read_palace_results(
        simulation_folder,
        filename,
        component.ports,
        is_temporary=str(simulation_folder) == temp_dir.name,
    )
    temp_dir.cleanup()
    return results


# if __name__ == "__main__":
#     import pyvista as pv

#     from gdsfactory.generic_tech import LAYER
#     from gdsfactory.technology.layer_stack import LayerLevel

#     # Example LayerStack similar to doi:10.1103/PRXQuantum.4.010314
#     layer_stack = LayerStack(
#         layers=dict(
#             substrate=LayerLevel(
#                 layer=LAYER.WAFER,
#                 thickness=500,
#                 zmin=0,
#                 material="Si",
#                 mesh_order=99,
#             ),
#             metal=LayerLevel(
#                 layer=LAYER.WG,
#                 thickness=200e-3,
#                 zmin=500,
#                 material="Nb",
#                 mesh_order=2,
#             ),
#         )
#     )
#     material_spec = {
#         "Si": {"relative_permittivity": 11.45, "relative_permeability": 1},
#         "Nb": {"relative_permittivity": inf},
#         "vacuum": {"relative_permittivity": 1, "relative_permeability": 1},
#     }

#     # Test capacitor
#     simulation_box = [[-200, -200], [200, 200]]
#     c = gf.Component("scattering_palace")
#     cap = c << interdigital_capacitor_enclosed(
#         metal_layer=LAYER.WG, gap_layer=LAYER.DEEPTRENCH, enclosure_box=simulation_box
#     )
#     c.add_ports(cap.ports)
#     # TODO ports to sides
#     substrate = gf.components.bbox(bbox=simulation_box, layer=LAYER.WAFER)
#     c << substrate
#     c.flatten()

#     results = run_scattering_simulation_palace(
#         c,
#         layer_stack=layer_stack,
#         driven_settings={
#             "MinFreq": 0.1,
#             "MaxFreq": 5,
#             "FreqStep": 2,
#         },
#         mesh_parameters=dict(
#             background_tag="vacuum",
#             background_padding=(0,) * 5 + (700,),
#             portnames=c.ports,
#             verbosity=1,
#             default_characteristic_length=200,
#             layer_portname_delimiter=(delimiter := "__"),
#             resolutions={
#                 "bw": {
#                     "resolution": 14,
#                 },
#                 "substrate": {
#                     "resolution": 50,
#                 },
#                 "vacuum": {
#                     "resolution": 120,
#                 },
#                 **{
#                     f"bw{delimiter}{port}_vacuum": {
#                         "resolution": 8,
#                     }
#                     for port in c.ports
#                 },
#             },
#         ),
#     )
#     print(results)

#     if results.field_file_location:
#         field = pv.read(results.field_file_location[0])
#         field.slice_orthogonal().plot(scalars="Ue", cmap="turbo")
