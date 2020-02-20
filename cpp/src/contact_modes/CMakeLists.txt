cmake_minimum_required(VERSION 3.2)

# Source files.
set(CONTACT_MODES_SOURCE
    # Geometry
    geometry/arrangements.cpp
    geometry/incidence_graph.cpp
    collision/collide_2d.cpp

    # Util
    backward.cpp
)

add_library(contact_modes SHARED ${CONTACT_MODES_SOURCE})

target_link_libraries(contact_modes dw)

# Python module.
pybind11_add_module(_contact_modes
    MODULE
    NO_EXTRAS
    pybind/module.cpp
    pybind/exportGeometry.cpp
    )

target_link_libraries(_contact_modes PRIVATE contact_modes pthread)
set_target_properties(_contact_modes
    PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../python/src/contact_modes"
    )
