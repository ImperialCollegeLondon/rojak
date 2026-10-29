"""Structural tests for :mod:`rojak.atmosphere.feature_composite` (phase 1, plan sections 1.2, 1.4 and 6).

These guard the shape of the subpackage rather than its numbers:

- **API scope:** the package namespace (for users) and ``extension`` (for extenders) export exactly the curated names,
  only re-export them from private modules, and do not overlap.
- **Import layering:** every module's runtime imports, read with :mod:`ast` (no import-linter, D9), point only to
  strictly lower layers of the subpackage (section 1.4, with ``_frame`` in layer 1 per D16), never to a namespace, never
  relatively, and only to the allowed parts of the rest of ``rojak``.
- **Fresh interpreters:** each module imports on its own, the namespaces do not pull in the orchestrator, CLI, plotting,
  data libraries or ``core.data``, and ``import rojak.atmosphere`` does not pull in the subpackage. The test session
  imports everything up front, which hides cycles that only appear for a particular import order, so these run in a
  subprocess.
"""

import ast
import functools
import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE: str = "rojak.atmosphere.feature_composite"
EXTENSION: str = f"{PACKAGE}.extension"

# Phase 1 exports. Later phases add NearestCentrePairing (package), EventCondition and orientation_from_flow
# (extension); adding an export means changing these lists, which makes it a deliberate decision.
PACKAGE_EXPORTS: tuple[str, ...] = (
    "composite_observations_about_features",
    "ExtremaPointFeature",
    "LabelledRegionPairing",
    "DistancePairing",
    "BinSpec",
    "ControlDesign",
    "impact_statistics",
    "bootstrap",
    "Column",
)
EXTENSION_EXPORTS: tuple[str, ...] = (
    "FeatureSource",
    "FeatureTabulator",
    "FeatureChunk",
    "ChunkInfo",
    "FeatureGeometry",
    "PairingStrategy",
)
NAMESPACE_EXPORTS: dict[str, tuple[str, ...]] = {PACKAGE: PACKAGE_EXPORTS, EXTENSION: EXTENSION_EXPORTS}

# Section 1.4, with _frame moved to layer 1 (D16). A module may import only strictly lower layers at runtime.
LAYERS: dict[str, int] = {
    "_columns": 0,
    "_feature": 1,
    "_frame": 1,
    "_point_features": 2,
    "_pairing": 2,
    "_binning": 2,
    "_controls": 2,
    "_statistics": 2,
    "_pipeline": 3,
    "__init__": 4,
    "extension": 4,
}
NAMESPACE_MODULES: frozenset[str] = frozenset({"__init__", "extension"})

# Runtime imports allowed from the rest of rojak (section 1.4). Each entry allows the module and anything below it.
ALLOWED_ROJAK_IMPORTS: tuple[str, ...] = (
    "rojak.utilities",
    "rojak.core.indexing",
    "rojak.core.geometric",
    "rojak.atmosphere.regions",
    "rojak.atmosphere.compositing",
    "rojak.atmosphere._compositing_kernels",
    "rojak.turbulence.metrics",
    "rojak.turbulence.calculations",
)
# Must never be loaded at runtime by importing a public namespace of the subpackage (section 1.4).
FORBIDDEN_AT_RUNTIME: tuple[str, ...] = (
    "rojak.orchestrator",
    "rojak.cli",
    "rojak.plot",
    "rojak.datalib",
    "rojak.core.data",
)
FRESH_INTERPRETER_TIMEOUT_S: float = 120.0


def _package_dir() -> Path:
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None
    assert spec.submodule_search_locations is not None
    return Path(next(iter(spec.submodule_search_locations)))


PACKAGE_DIR: Path = _package_dir()
DISCOVERED_MODULES: tuple[str, ...] = tuple(sorted(path.stem for path in PACKAGE_DIR.glob("*.py")))
# The planned modules plus whatever is on disk, so a module that is missing or new is still covered.
ALL_MODULES: tuple[str, ...] = tuple(sorted(set(LAYERS) | set(DISCOVERED_MODULES)))


def _dotted(module: str) -> str:
    return PACKAGE if module == "__init__" else f"{PACKAGE}.{module}"


# ---------------------------------------------------------------------------------------------------------------------
# Runtime-import collection and the layering rules, as pure functions of the source text
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Import:
    line: int
    module: str  # ``X`` in ``import X`` or ``from X import ...``; empty for ``from . import x``
    names: tuple[str, ...]  # names after ``import`` in a from-import; empty for a plain import
    level: int  # number of leading dots of a relative import


def _is_type_checking_guard(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` however it is spelled (``typing.TYPE_CHECKING``, ``t.TYPE_CHECKING``)."""
    match test:
        case ast.Name(id="TYPE_CHECKING") | ast.Attribute(attr="TYPE_CHECKING"):
            return True
        case _:
            return False


def _runtime_imports(source: str) -> list[_Import]:
    """Every import that executes at runtime, including the ``else`` of a ``TYPE_CHECKING`` guard and lazy imports."""
    found: list[_Import] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.extend(_Import(node.lineno, alias.name, (), 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            found.append(_Import(node.lineno, node.module or "", names, node.level))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return found


def _subpackage_targets(imported: _Import) -> set[str]:
    """Modules of the subpackage an absolute import executes (``"__init__"`` for the package namespace)."""
    if imported.level != 0:
        return set()
    if imported.module == PACKAGE:
        # ``from PACKAGE import _columns`` imports the submodule; ``import PACKAGE`` or ``from PACKAGE import Column``
        # reads from the namespace itself, which is the classic cycle while the package initialises
        submodules = {name for name in imported.names if name in ALL_MODULES and name != "__init__"}
        reads_namespace = not imported.names or len(submodules) < len(imported.names)
        return submodules | ({"__init__"} if reads_namespace else set())
    if imported.module.startswith(f"{PACKAGE}."):
        return {imported.module.removeprefix(f"{PACKAGE}.").split(".")[0]}
    return set()


def _is_allowed_rojak_import(dotted: str) -> bool:
    return any(dotted == allowed or dotted.startswith(f"{allowed}.") for allowed in ALLOWED_ROJAK_IMPORTS)


def _is_rest_of_rojak(dotted: str) -> bool:
    in_rojak = dotted == "rojak" or dotted.startswith("rojak.")
    in_subpackage = dotted == PACKAGE or dotted.startswith(f"{PACKAGE}.")
    return in_rojak and not in_subpackage


def _relative_import_violations(module: str, source: str) -> list[str]:
    return [
        f"{module}.py:{imported.line}: relative import {'.' * imported.level}{imported.module}"
        for imported in _runtime_imports(source)
        if imported.level != 0
    ]


def _layer_violations(module: str, source: str) -> list[str]:
    own_layer = LAYERS[module]
    violations: list[str] = []
    for imported in _runtime_imports(source):
        for target in sorted(_subpackage_targets(imported)):
            if target not in LAYERS:
                violations.append(f"{module}.py:{imported.line}: imports {target}, which has no layer")
            elif LAYERS[target] >= own_layer:
                violations.append(
                    f"{module}.py:{imported.line}: layer {own_layer} imports {target} (layer {LAYERS[target]})"
                )
    return violations


def _namespace_import_violations(module: str, source: str) -> list[str]:
    return [
        f"{module}.py:{imported.line}: imports the public namespace {_dotted(target)}"
        for imported in _runtime_imports(source)
        for target in sorted(_subpackage_targets(imported) & NAMESPACE_MODULES)
    ]


def _rest_of_rojak_violations(module: str, source: str) -> list[str]:
    violations: list[str] = []
    for imported in _runtime_imports(source):
        if imported.level != 0 or not _is_rest_of_rojak(imported.module):
            continue
        # ``from rojak.atmosphere import regions`` imports ``rojak.atmosphere.regions``
        allowed = _is_allowed_rojak_import(imported.module) or (
            bool(imported.names)
            and all(_is_allowed_rojak_import(f"{imported.module}.{name}") for name in imported.names)
        )
        if not allowed:
            names = f" ({', '.join(imported.names)})" if imported.names else ""
            violations.append(f"{module}.py:{imported.line}: imports {imported.module}{names}, not on the allowed list")
    return violations


def _source(module: str) -> str:
    path = PACKAGE_DIR / f"{module}.py"
    if not path.is_file():
        pytest.fail(f"{_dotted(module)} does not exist ({path})")
    return path.read_text(encoding="utf-8")


def _report(violations: list[str]) -> str:
    return "\n".join(violations)


# ---------------------------------------------------------------------------------------------------------------------
# Fresh interpreters
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _FreshImport:
    returncode: int
    stderr: str
    modules: frozenset[str]


@functools.cache
def _fresh_import(dotted: str) -> _FreshImport:
    """Import ``dotted`` in a new interpreter and return what ended up in ``sys.modules``.

    Cached, so a namespace imported by several tests in the same worker costs one subprocess.
    """
    code = f"import json, sys\nimport {dotted}\nprint(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=FRESH_INTERPRETER_TIMEOUT_S,
    )
    modules: frozenset[str] = frozenset()
    if result.returncode == 0:
        modules = frozenset(json.loads(result.stdout.strip().splitlines()[-1]))
    return _FreshImport(result.returncode, result.stderr, modules)


def _loaded_forbidden(modules: frozenset[str]) -> list[str]:
    return sorted(
        name
        for name in modules
        if any(name == forbidden or name.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_AT_RUNTIME)
    )


# ---------------------------------------------------------------------------------------------------------------------
# API scope
# ---------------------------------------------------------------------------------------------------------------------


def _exported(namespace: str) -> list[str]:
    module = importlib.import_module(namespace)
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{namespace} defines no __all__"
    return list(exported)


def _submodule_attributes(module: ModuleType) -> set[str]:
    """Attributes Python itself sets on a package when one of its submodules is imported."""
    return {
        name
        for name in dir(module)
        if isinstance(attribute := getattr(module, name), ModuleType)
        and attribute.__name__ == f"{module.__name__}.{name}"
    }


class TestPublicApiScope:
    @pytest.mark.parametrize(("namespace", "expected"), list(NAMESPACE_EXPORTS.items()))
    def test_all_equals_the_curated_list(self, namespace: str, expected: tuple[str, ...]) -> None:
        assert set(_exported(namespace)) == set(expected)

    @pytest.mark.parametrize("namespace", list(NAMESPACE_EXPORTS))
    def test_all_has_no_duplicates(self, namespace: str) -> None:
        exported = _exported(namespace)
        assert len(exported) == len(set(exported))

    @pytest.mark.parametrize(
        ("namespace", "name"),
        [(namespace, name) for namespace, names in NAMESPACE_EXPORTS.items() for name in names],
    )
    def test_every_exported_name_imports(self, namespace: str, name: str) -> None:
        module = importlib.import_module(namespace)
        assert hasattr(module, name), f"cannot import {name} from {namespace}"

    @pytest.mark.parametrize("namespace", list(NAMESPACE_EXPORTS))
    def test_namespace_has_no_other_public_names(self, namespace: str) -> None:
        module = importlib.import_module(namespace)
        public = {name for name in dir(module) if not name.startswith("_")}
        unexpected = public - set(_exported(namespace)) - _submodule_attributes(module)
        assert unexpected == set()

    def test_namespaces_do_not_overlap(self) -> None:
        assert set(_exported(PACKAGE)) & set(_exported(EXTENSION)) == set()

    def test_curated_lists_do_not_overlap(self) -> None:
        assert set(PACKAGE_EXPORTS) & set(EXTENSION_EXPORTS) == set()

    def test_extension_is_not_exported_from_the_package(self) -> None:
        assert "extension" not in _exported(PACKAGE)

    @pytest.mark.parametrize(
        ("namespace", "name"),
        [(namespace, name) for namespace, names in NAMESPACE_EXPORTS.items() for name in names],
    )
    def test_exported_object_is_defined_in_a_private_module(self, namespace: str, name: str) -> None:
        exported = getattr(importlib.import_module(namespace), name)
        defining_module: str = getattr(exported, "__module__", "")
        parent, _, leaf = defining_module.rpartition(".")
        assert parent == PACKAGE, f"{name} is defined in {defining_module!r}, outside {PACKAGE}"
        assert leaf.startswith("_"), f"{name} is defined in the public module {defining_module!r}"


# ---------------------------------------------------------------------------------------------------------------------
# Import layering (static, ast)
# ---------------------------------------------------------------------------------------------------------------------


class TestRuntimeImportCollector:
    """The ast helpers above, checked on hand-written sources, so the layering tests can be trusted."""

    def test_records_plain_and_from_imports_with_line_numbers(self) -> None:
        source = "import numpy as np\nfrom rojak.core.indexing import nearest_index, apply_nan_mask\n"
        assert _runtime_imports(source) == [
            _Import(1, "numpy", (), 0),
            _Import(2, "rojak.core.indexing", ("nearest_index", "apply_nan_mask"), 0),
        ]

    @pytest.mark.parametrize(
        "guard",
        ["from typing import TYPE_CHECKING\nif TYPE_CHECKING:", "import typing\nif typing.TYPE_CHECKING:"],
    )
    def test_skips_type_checking_blocks_however_spelled(self, guard: str) -> None:
        source = f"{guard}\n    from rojak.orchestrator.configuration import SpatialDomain\n"
        assert [imported.module for imported in _runtime_imports(source)] == ["typing"]

    def test_keeps_the_else_branch_of_a_type_checking_guard(self) -> None:
        source = textwrap.dedent(
            """
            if TYPE_CHECKING:
                import pandas
            else:
                import rojak.plot
            """
        )
        assert [imported.module for imported in _runtime_imports(source)] == ["rojak.plot"]

    def test_keeps_lazy_imports_inside_functions(self) -> None:
        source = "def f():\n    from rojak.cli import main\n    return main\n"
        assert [imported.module for imported in _runtime_imports(source)] == ["rojak.cli"]

    def test_records_relative_import_level(self) -> None:
        source = "from . import _columns\nfrom .._compositing_kernels import orthographic_projection\n"
        assert [(imported.module, imported.level) for imported in _runtime_imports(source)] == [
            ("", 1),
            ("_compositing_kernels", 2),
        ]


class TestLayeringRules:
    """The rules themselves, checked on hand-written sources."""

    @pytest.mark.parametrize(
        ("module", "source"),
        [
            ("_pairing", f"from {PACKAGE}._frame import project_pairs"),
            ("_pairing", f"from {PACKAGE}._columns import Column"),
            ("_pipeline", f"from {PACKAGE}._binning import BinSpec"),
            ("__init__", f"from {PACKAGE}._pipeline import composite_observations_about_features"),
            ("extension", f"from {PACKAGE}._feature import FeatureSource"),
            ("_columns", f"if TYPE_CHECKING:\n    from {PACKAGE}._feature import FeatureSource"),
        ],
    )
    def test_accepts_imports_of_strictly_lower_layers(self, module: str, source: str) -> None:
        assert _layer_violations(module, source) == []

    @pytest.mark.parametrize(
        ("module", "source"),
        [
            ("_columns", f"from {PACKAGE}._feature import FeatureSource"),
            ("_feature", f"from {PACKAGE}._frame import project_pairs"),
            ("_pairing", f"from {PACKAGE}._binning import BinSpec"),
            ("_statistics", f"import {PACKAGE}._pipeline"),
            ("__init__", f"from {PACKAGE} import extension"),
            ("extension", f"from {PACKAGE} import Column"),
            ("_pipeline", f"from {PACKAGE}._unknown import thing"),
        ],
    )
    def test_rejects_same_higher_or_unknown_layer(self, module: str, source: str) -> None:
        violations = _layer_violations(module, source)
        assert len(violations) == 1
        assert violations[0].startswith(f"{module}.py:1:")

    @pytest.mark.parametrize(
        "source",
        [f"from {PACKAGE} import Column", f"import {PACKAGE}", f"from {PACKAGE}.extension import FeatureSource"],
    )
    def test_rejects_namespace_imports(self, source: str) -> None:
        assert len(_namespace_import_violations("_pairing", source)) == 1

    @pytest.mark.parametrize(
        "source",
        [
            "from rojak.utilities.types import is_xr_data_array",
            "import rojak.utilities._compat",
            "from rojak.atmosphere import regions",
            "from rojak.atmosphere._compositing_kernels import orthographic_projection",
            "from rojak.turbulence.metrics import ContingencyTable",
            "import numpy as np",
        ],
    )
    def test_accepts_allowed_imports_from_the_rest_of_rojak(self, source: str) -> None:
        assert _rest_of_rojak_violations("_pairing", source) == []

    @pytest.mark.parametrize(
        "source",
        [
            "from rojak.orchestrator.configuration import DiagnosticValidationCondition",
            "from rojak.core.data import CATData",
            "from rojak.core import geometric, data",
            "from rojak.atmosphere import regions, contrails",
            "import rojak.plot",
            "from rojak.turbulence.diagnostic import DiagnosticFactory",
        ],
    )
    def test_rejects_other_imports_from_the_rest_of_rojak(self, source: str) -> None:
        assert len(_rest_of_rojak_violations("_pairing", source)) == 1


class TestImportLayering:
    def test_every_module_on_disk_has_a_layer(self) -> None:
        assert set(DISCOVERED_MODULES) - set(LAYERS) == set()

    @pytest.mark.parametrize("module", ALL_MODULES)
    def test_imports_within_subpackage_point_to_strictly_lower_layers(self, module: str) -> None:
        if module not in LAYERS:
            pytest.fail(f"{_dotted(module)} has no layer in section 1.4")
        violations = _layer_violations(module, _source(module))
        assert violations == [], _report(violations)

    @pytest.mark.parametrize("module", [module for module in ALL_MODULES if module not in NAMESPACE_MODULES])
    def test_private_module_does_not_import_a_public_namespace(self, module: str) -> None:
        violations = _namespace_import_violations(module, _source(module))
        assert violations == [], _report(violations)

    @pytest.mark.parametrize("module", ALL_MODULES)
    def test_has_no_relative_imports(self, module: str) -> None:
        violations = _relative_import_violations(module, _source(module))
        assert violations == [], _report(violations)

    @pytest.mark.parametrize("module", ALL_MODULES)
    def test_imports_from_rest_of_rojak_are_on_the_allowed_list(self, module: str) -> None:
        violations = _rest_of_rojak_violations(module, _source(module))
        assert violations == [], _report(violations)


# ---------------------------------------------------------------------------------------------------------------------
# Fresh interpreters (one subprocess per module)
# ---------------------------------------------------------------------------------------------------------------------


class TestIsolatedImport:
    @pytest.mark.parametrize("module", ALL_MODULES)
    def test_module_imports_in_a_fresh_interpreter(self, module: str) -> None:
        result = _fresh_import(_dotted(module))
        assert result.returncode == 0, result.stderr


class TestNoOrchestratorAtRuntime:
    @pytest.mark.parametrize("namespace", list(NAMESPACE_EXPORTS))
    def test_namespace_does_not_load_orchestrator_cli_plot_datalib_or_core_data(self, namespace: str) -> None:
        result = _fresh_import(namespace)
        assert result.returncode == 0, result.stderr
        assert _loaded_forbidden(result.modules) == []

    def test_package_does_not_load_extension(self) -> None:
        result = _fresh_import(PACKAGE)
        assert result.returncode == 0, result.stderr
        assert EXTENSION not in result.modules


class TestAtmospherePackageIndependence:
    def test_importing_rojak_atmosphere_does_not_load_the_subpackage(self) -> None:
        result = _fresh_import("rojak.atmosphere")
        assert result.returncode == 0, result.stderr
        assert PACKAGE not in result.modules
