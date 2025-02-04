# simple std vector of eigen vectors
pybind11_add_module(pyvector MODULE pyvector.cpp)
target_link_libraries(pyvector PRIVATE
    Eigen3::Eigen
)

install(TARGETS pyvector DESTINATION .)

pybind11_add_module(pygsaligner MODULE pygsaligner.cpp)
target_link_libraries(pygsaligner PRIVATE
    pybind11::module
    Eigen3::Eigen
    registration
    tools
)

install(TARGETS pygsaligner DESTINATION .)