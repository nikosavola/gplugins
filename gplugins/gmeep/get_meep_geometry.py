from __future__ import annotations

import gdsfactory as gf
import meep as mp
import numpy as np
from kfactory import LayerEnum
from typing import cast
import shapely
from gdsfactory.pdk import get_layer_stack, get_layer, get_layer_name
from gdsfactory.technology import LayerStack
from gdsfactory.technology import DerivedLayer, LayerStack, LayerViews, LogicalLayer
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpecs

from gplugins.common.utils.parse_layer_stack import order_layer_stack
from gplugins.gmeep.get_material import get_material


def get_meep_geometry_from_component(
    component: ComponentSpec,
    layer_stack: LayerStack | None = None,
    material_name_to_meep: dict[str, str | float] | None = None,
    wavelength: float = 1.55,
    is_3d: bool = False,
    dispersive: bool = False,
    exclude_layers: LayerSpecs | None = None,
    **kwargs,
) -> list[mp.GeometricObject]:
    """Returns Meep geometry from a gdsfactory component.

    Args:
        component: gdsfactory component.
        layer_stack: for material layers.
        material_name_to_meep: maps layer_stack name to meep material name.
        wavelength: in um.
        is_3d: renders in 3D.
        dispersive: add dispersion.
        kwargs: settings.
    """
    component = gf.get_component(component=component, **kwargs)
    polygons_per_layer = component.get_polygons_points(merge=True)


    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()
    component_with_booleans = layer_stack.get_component_with_derived_layers(component)

    geometry = []
    exclude_layers = exclude_layers or []
    layer_to_polygons = component_with_booleans.get_polygons_points()

    # ordered_layer_stack_keys = order_layer_stack(layer_stack)[::-1]

    for level in layer_stack.layers.values():
        layer = level.layer

        if isinstance(layer, LogicalLayer):
            layer_tuple = gf.get_layer_tuple(layer.layer)
        elif isinstance(layer, DerivedLayer):
            layer_tuple = gf.get_layer_tuple(level.derived_layer.layer)
        elif isinstance(layer, tuple):
            # Handle plain tuple layers directly
            layer_tuple = layer
        else:
            raise ValueError(f"Layer {layer!r} is not a DerivedLayer, LogicalLayer, or tuple")

        layer_index = int(get_layer(layer_tuple))

        if layer_index in exclude_layers:
            continue

        if layer_index not in polygons_per_layer:
            continue

        zmin = level.zmin
        zmin_um = layer_to_zmin[layer] if is_3d else 0
        if zmin is not None:
            has_polygons = True
            polygons = polygons_per_layer[layer_index]
            height = level.thickness
            for polygon in polygons:
                p = shapely.geometry.Polygon(polygon)
                vertices = [mp.Vector3(p[0], p[1], zmin_um) for p in polygon]
                material_name = layer_to_material[layer]

                if material_name:
                    material = get_material(
                        name=material_name,
                        dispersive=dispersive,
                        material_name_to_meep=material_name_to_meep,
                        wavelength=wavelength,
                    )
                    geometry.append(
                        mp.Prism(
                            vertices=vertices,
                            height=height,
                            sidewall_angle=np.pi * layer_to_sidewall_angle[layer] / 180
                            if is_3d
                            else 0,
                            material=material,
                            # center=center
                        )
                    )

    return geometry


def get_meep_geometry_from_cross_section(
    cross_section: CrossSectionSpec,
    extension_length: float | None = None,
    layer_stack: LayerStack | None = None,
    material_name_to_meep: dict[str, str | float] | None = None,
    wavelength: float = 1.55,
    dispersive: bool = False,
    **kwargs,
) -> list[mp.GeometricObject]:
    x = gf.get_cross_section(cross_section=cross_section, **kwargs)

    x_sections = [
        gf.Section(offset=x.offset, layer=x.layer, width=x.width),
        *x.sections,
    ]

    layer_stack = layer_stack or get_layer_stack()

    layer_to_thickness = layer_stack.get_layer_to_thickness()
    layer_to_material = layer_stack.get_layer_to_material()
    layer_to_zmin = layer_stack.get_layer_to_zmin()
    layer_to_sidewall_angle = layer_stack.get_layer_to_sidewall_angle()

    geometry = []
    for section in x_sections:
        print(f"section: {section}")
        layer = gf.get_layer(section.layer)

        if layer in layer_to_thickness and layer in layer_to_material:
            height = layer_to_thickness[layer]
            width = section.width
            offset = section.offset

            zmin_um = layer_to_zmin[layer] + (0 if height > 0 else -height)
            # center = mp.Vector3(0, 0, (zmin_um + height) / 2)

            material_name = layer_to_material[layer]
            material = get_material(
                name=material_name,
                dispersive=dispersive,
                material_name_to_meep=material_name_to_meep,
                wavelength=wavelength,
            )
            index = material.epsilon(1 / wavelength)[0, 0] ** 0.5
            print(f"add {material_name!r} layer with index {index}")
            # Don't need to use prism unless using sidewall angles
            if layer in layer_to_sidewall_angle:
                # If using a prism, all dimensions need to be finite
                xspan = extension_length or 1
                p = mp.Prism(
                    vertices=[
                        mp.Vector3(x=-xspan / 2, y=-width / 2, z=zmin_um),
                        mp.Vector3(x=-xspan / 2, y=width / 2, z=zmin_um),
                        mp.Vector3(x=xspan / 2, y=width / 2, z=zmin_um),
                        mp.Vector3(x=xspan / 2, y=-width / 2, z=zmin_um),
                    ],
                    height=height,
                    center=mp.Vector3(y=offset, z=height / 2 + zmin_um),
                    sidewall_angle=np.deg2rad(layer_to_sidewall_angle[layer]),
                    material=material,
                )
                geometry.append(p)

            else:
                xspan = extension_length or mp.inf
                geometry.append(
                    mp.Block(
                        size=mp.Vector3(xspan, width, height),
                        material=material,
                        center=mp.Vector3(y=offset, z=height / 2 + zmin_um),
                    )
                )
    return geometry


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import gplugins.gmeep as gm

    c = gf.components.straight()
    sp = gm.write_sparameters_meep(
        c, run=False, ymargin_top=3, ymargin_bot=3, is_3d=True
    )
    plt.show()
    # c.show()
