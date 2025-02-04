#include <torch/extension.h>

#include "gsaligner/aligner.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("align_geometric",)
  m.def("RegistrationGeomPhoto", &RegistrationGeomPhoto);
}