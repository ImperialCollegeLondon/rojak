import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, Mapping, assert_never

import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from rich.progress import track

from rojak.core.derivatives import (
    CartesianDimension,
    GradientMode,
    VelocityDerivative,
    spatial_gradient,
    spatial_laplacian,
)
from rojak.orchestrator.configuration import (
    TurbulenceDiagnostics,
    TurbulenceSeverity,
    TurbulenceThresholdMode,
    TurbulenceThresholds,
)
from rojak.turbulence.analysis import (
    DiagnosticHistogramDistribution,
    TransformToEDR,
    TurbulenceIntensityThresholds,
    TurbulenceProbabilityBySeverity,
)
from rojak.turbulence.calculations import (
    GRAVITATIONAL_ACCELERATION,
    absolute_vorticity,
    altitude_derivative_on_pressure_level,
    angles_gradient,
    coriolis_parameter,
    latitudinal_derivative,
    magnitude_of_geospatial_gradient,
    magnitude_of_vector,
    vertical_wind_shear,
    wind_direction,
    wind_speed,
)

if TYPE_CHECKING:
    from rojak.core.data import CATData
    from rojak.core.derivatives import SpatialGradientKeys
    from rojak.orchestrator.configuration import TurbulenceSeverity
    from rojak.turbulence.analysis import HistogramData
    from rojak.utilities.types import DiagnosticName, DistributionParameters

logger = logging.getLogger(__name__)


class Diagnostic(ABC):
    _name: "DiagnosticName"
    _computed_value: None | xr.DataArray = None

    def __init__(self, name: "DiagnosticName") -> None:
        self._name = name

    @abstractmethod
    def _compute(self) -> xr.DataArray:
        pass

    # TODO: TEEST
    @property
    def name(self) -> "DiagnosticName":
        return self._name

    # TODO: TEEST
    @property
    def computed_value(self) -> xr.DataArray:
        if self._computed_value is None:
            self._computed_value = self._compute()
        return self._computed_value


class Frontogenesis3D(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _potential_temperature: xr.DataArray
    _geopotential: xr.DataArray
    _divergence: xr.DataArray
    _du_dx: xr.DataArray
    _dv_dx: xr.DataArray
    _du_dy: xr.DataArray
    _dv_dy: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        potential_temperature: xr.DataArray,
        geopotential: xr.DataArray,
        divergence: xr.DataArray,
        vector_derivatives: dict[VelocityDerivative, xr.DataArray],
    ) -> None:
        super().__init__("F3D")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._potential_temperature = potential_temperature
        self._geopotential = geopotential
        self._divergence = divergence
        self._du_dx = vector_derivatives[VelocityDerivative.DU_DX]
        self._dv_dx = vector_derivatives[VelocityDerivative.DV_DX]
        self._du_dy = vector_derivatives[VelocityDerivative.DU_DY]
        self._dv_dy = vector_derivatives[VelocityDerivative.DV_DY]

    def x_component(self, dtheta_dx: xr.DataArray, dtheta_dy: xr.DataArray) -> xr.DataArray:
        return dtheta_dx * (self._du_dx * dtheta_dx + self._dv_dx * dtheta_dy)

    def y_component(self, dtheta_dx: xr.DataArray, dtheta_dy: xr.DataArray) -> xr.DataArray:
        return dtheta_dy * (self._du_dy * dtheta_dx + self._dv_dy * dtheta_dy)

    def z_component(self, dtheta_dx: xr.DataArray, dtheta_dy: xr.DataArray, dtheta_dz: xr.DataArray) -> xr.DataArray:
        du_dz: xr.DataArray = altitude_derivative_on_pressure_level(self._u_wind, self._geopotential)
        dv_dz: xr.DataArray = altitude_derivative_on_pressure_level(self._v_wind, self._geopotential)
        return dtheta_dz * (du_dz * dtheta_dx + dv_dz * dtheta_dy - self._divergence * dtheta_dz)

    # TODO: TEEST
    def _compute(self) -> xr.DataArray:
        r"""
        .. math:: \mathbf{F} = - \frac{1}{|\nabla\theta|} &\left[\frac{ \partial \theta }{ \partial x }
        \left(  \frac{ \partial u }{ \partial x } \frac{ \partial \theta }{ \partial x } +
        \frac{ \partial v }{ \partial x } \frac{ \partial \theta }{ \partial y } \right) \right.
        + \left.  \frac{ \partial \theta }{ \partial y } \left(  \frac{ \partial u }{ \partial y }
        \frac{ \partial \theta }{ \partial x } + \frac{ \partial v }{ \partial y }
        \frac{ \partial \theta }{ \partial y } \right) \right. \\
        &+ \left. \frac{ \partial \theta }{ \partial z } \left(  \frac{ \partial u }{ \partial z }
        \frac{ \partial \theta }{ \partial x } + \frac{ \partial v }{ \partial z } \frac{ \partial \theta }{ \partial y}
        - \delta \frac{ \partial \theta }{ \partial z }\right) \right]
        """
        theta_horz_gradient = spatial_gradient(self._potential_temperature, "deg", GradientMode.GEOSPATIAL)
        dtheta_dx = theta_horz_gradient["dfdx"]
        dtheta_dy = theta_horz_gradient["dfdy"]
        dtheta_dz = altitude_derivative_on_pressure_level(self._potential_temperature, self._geopotential)

        inverse_mag_grad_theta: xr.DataArray = 1 / np.sqrt(
            dtheta_dx * dtheta_dx + dtheta_dy * dtheta_dy + dtheta_dz * dtheta_dz
        )  # pyright: ignore[reportAssignmentType]
        # If potential field has no changes, then there will be a division by zero
        inverse_mag_grad_theta = inverse_mag_grad_theta.fillna(0)

        return inverse_mag_grad_theta * (
            self.x_component(dtheta_dx, dtheta_dy)
            + self.y_component(dtheta_dx, dtheta_dy)
            + self.z_component(dtheta_dx, dtheta_dy, dtheta_dz)
        )


class Frontogenesis2D(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _potential_temperature: xr.DataArray
    _geopotential: xr.DataArray
    _du_dx: xr.DataArray
    _dv_dx: xr.DataArray
    _du_dy: xr.DataArray
    _dv_dy: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        potential_temperature: xr.DataArray,
        geopotential: xr.DataArray,
        vector_derivatives: dict[VelocityDerivative, xr.DataArray],
    ) -> None:
        super().__init__("F2D")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._potential_temperature = potential_temperature
        self._geopotential = geopotential
        self._du_dx = vector_derivatives[VelocityDerivative.DU_DX]
        self._dv_dx = vector_derivatives[VelocityDerivative.DV_DX]
        self._du_dy = vector_derivatives[VelocityDerivative.DU_DY]
        self._dv_dy = vector_derivatives[VelocityDerivative.DV_DY]

    def _compute(self) -> xr.DataArray:
        r"""
        .. math:: F = - \frac{1}{\left| \nabla_{p} \theta \right|} \left[\left( \frac{ \partial \theta }{ \partial x }
        \right)^{2} \frac{ \partial u }{ \partial x } + \frac{ \partial \theta }{ \partial y }
        \frac{ \partial \theta }{ \partial x }\frac{ \partial v }{ \partial x } + \frac{ \partial \theta }{ \partial x }
        \frac{ \partial \theta }{ \partial y }\frac{ \partial u }{ \partial y } +
        \left( \frac{ \partial \theta }{ \partial y }  \right)^{2} \frac{ \partial v }{ \partial y }\right]
        """
        dtheta: dict[SpatialGradientKeys, xr.DataArray] = spatial_gradient(
            self._potential_temperature, "deg", GradientMode.GEOSPATIAL
        )
        inverse_mag_grad_theta: xr.DataArray = -1 / magnitude_of_vector(dtheta["dfdx"], dtheta["dfdy"])
        # If potential field has no changes, then there will be a division by zero
        inverse_mag_grad_theta = inverse_mag_grad_theta.fillna(0)

        return inverse_mag_grad_theta * (
            dtheta["dfdx"] * dtheta["dfdx"] * self._du_dx
            + dtheta["dfdy"] * dtheta["dfdy"] * self._dv_dy
            + dtheta["dfdx"] * dtheta["dfdy"] * self._dv_dy
            + dtheta["dfdx"] * dtheta["dfdy"] * self._du_dy
        )


class HorizontalTemperatureGradient(Diagnostic):
    _temperature: xr.DataArray

    def __init__(self, temperature: xr.DataArray) -> None:
        super().__init__("Horizontal Temperature Gradient")
        self._temperature = temperature

    def _compute(self) -> xr.DataArray:
        return magnitude_of_geospatial_gradient(self._temperature)


class Endlich(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _wind_direction: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray) -> None:
        super().__init__("Endlich")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._wind_direction = wind_direction(u_wind, v_wind)
        self._geopotential = geopotential

    def _compute(self) -> xr.DataArray:
        values_in_z_axis: np.ndarray = self._u_wind["pressure_level"].data
        z_axis: int = self._u_wind.get_axis_num("pressure_level")

        if is_dask_collection(self._u_wind):
            d_direction_d_p: xr.DataArray = xr.apply_ufunc(
                angles_gradient,
                self._wind_direction,
                kwargs={"coord_values": values_in_z_axis, "target_axis": z_axis},
                dask="parallelized",
                output_dtypes=[np.float32],
            ).compute()
        else:
            d_direction_d_p_values: np.ndarray = angles_gradient(
                self._wind_direction.values, z_axis, coord_values=values_in_z_axis
            )
            d_direction_d_p: xr.DataArray = self._wind_direction.copy(data=d_direction_d_p_values)

        d_direction_d_z: xr.DataArray = altitude_derivative_on_pressure_level(d_direction_d_p, self._geopotential)
        speed: xr.DataArray = wind_speed(self._u_wind, self._v_wind)
        return speed * np.abs(d_direction_d_z)


class TurbulenceIndex1(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray
    _total_deformation: xr.DataArray

    def __init__(
        self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray, total_deformation: xr.DataArray
    ) -> None:
        super().__init__("TI1")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential
        self._total_deformation = total_deformation

    def _compute(self) -> xr.DataArray:
        vws: xr.DataArray = vertical_wind_shear(self._u_wind, self._v_wind, geopotential=self._geopotential)
        return vws * self._total_deformation


class TurbulenceIndex2(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _du_dx: xr.DataArray
    _dv_dy: xr.DataArray
    _geopotential: xr.DataArray
    _total_deformation: xr.DataArray
    _divergence: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        vector_derivatives: dict[VelocityDerivative, xr.DataArray],
        geopotential: xr.DataArray,
        total_deformation: xr.DataArray,
        divergence: xr.DataArray,
    ) -> None:
        super().__init__("TI2")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._du_dx = vector_derivatives[VelocityDerivative.DU_DX]
        self._dv_dy = vector_derivatives[VelocityDerivative.DV_DY]
        self._geopotential = geopotential
        self._total_deformation = total_deformation
        self._divergence = divergence

    def _compute(self) -> xr.DataArray:
        vws: xr.DataArray = vertical_wind_shear(self._u_wind, self._v_wind, geopotential=self._geopotential)
        convergence: xr.DataArray = -self._divergence
        return vws * (self._total_deformation + convergence)


class Ncsu1(Diagnostic):
    RI_THRESHOLD: float = 1e-5
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _du_dx: xr.DataArray
    _dv_dy: xr.DataArray
    _ri: xr.DataArray
    _vorticity: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        ri: xr.DataArray,
        vector_derivatives: dict[VelocityDerivative, xr.DataArray],
        vorticity: xr.DataArray,
    ) -> None:
        super().__init__("NCSU1")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._du_dx = vector_derivatives[VelocityDerivative.DU_DX]
        self._dv_dy = vector_derivatives[VelocityDerivative.DV_DY]
        self._ri = xr.where(ri > self.RI_THRESHOLD, ri, self.RI_THRESHOLD)
        self._vorticity = vorticity

    def _compute(self) -> xr.DataArray:
        vorticity_term: xr.DataArray = magnitude_of_geospatial_gradient(self._vorticity)
        advection_term: xr.DataArray = self._u_wind * self._du_dx + self._v_wind * self._dv_dy
        advection_term = xr.where(advection_term > 0, advection_term, 0)
        return (vorticity_term * advection_term) / self._ri


class ColsonPanofsky(Diagnostic):
    _RI_CRIT: float = 0.5
    _length_scale: xr.DataArray
    _richardson_term: xr.DataArray
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        altitude: xr.DataArray,
        richardson_number: xr.DataArray,
        geopotential: xr.DataArray,
    ) -> None:
        super().__init__("Colson Panofsky")
        # !!! IMPT: Reduces dimension by 1
        # Want the minuend (i.e. label=upper) as we're looking at the distance between grid points (delta z)
        self._length_scale = altitude.diff("pressure_level", label="upper")
        new_pressure_level_coord = self._length_scale["pressure_level"]
        self._richardson_term = 1 - (richardson_number.sel(pressure_level=new_pressure_level_coord) / self._RI_CRIT)
        self._u_wind = u_wind.sel(pressure_level=new_pressure_level_coord)
        self._v_wind = v_wind.sel(pressure_level=new_pressure_level_coord)
        self._geopotential = geopotential.sel(pressure_level=new_pressure_level_coord)

    def _compute(self) -> xr.DataArray:
        vws: xr.DataArray = vertical_wind_shear(
            self._u_wind, self._v_wind, geopotential=self._geopotential, is_abs_velocities=True, is_vws_squared=True
        )
        return (self._length_scale * self._length_scale) * vws * self._richardson_term


class UBF(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray
    _coriolis_parameter: xr.DataArray
    _vorticity: xr.DataArray
    _jacobian: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        geopotential: xr.DataArray,
        vorticity: xr.DataArray,
        jacobian: xr.DataArray,
    ) -> None:
        super().__init__("UBF")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential
        self._coriolis_parameter = coriolis_parameter(u_wind["latitude"])
        # https://en.wikipedia.org/wiki/Rossby_parameter
        self._vorticity = vorticity
        self._jacobian = jacobian

    # Appears to work IF AND ONLY IF processes=False
    def _compute(self) -> xr.DataArray:
        coriolis_vorticity_term: xr.DataArray = self._coriolis_parameter * self._vorticity.metpy.dequantify()
        coriolis_deriv: xr.DataArray = latitudinal_derivative(self._coriolis_parameter)
        inertial_terms: xr.DataArray = coriolis_vorticity_term + 2 * self._jacobian
        mass_term: xr.DataArray = spatial_laplacian(self._geopotential, "deg", GradientMode.GEOSPATIAL)  # pyright: ignore[reportAssignmentType]

        return np.abs(mass_term + inertial_terms - coriolis_deriv * self._u_wind)  # pyright: ignore[reportReturnType]


class BruntVaisalaFrequency(Diagnostic):
    _potential_temperature: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(self, potential_temperature: xr.DataArray, geopotential: xr.DataArray) -> None:
        super().__init__("Brunt Vaisala Frequency")
        self._potential_temperature = potential_temperature
        self._geopotential = geopotential

    def _compute(self) -> xr.DataArray:
        d_potential_temperature_dz: xr.DataArray = altitude_derivative_on_pressure_level(
            self._potential_temperature, self._geopotential
        )
        # Negative value is to ensure percentile picks up the unstable values
        return -((GRAVITATIONAL_ACCELERATION / self._potential_temperature) * d_potential_temperature_dz)


class VerticalWindShear(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray) -> None:
        super().__init__("Vertical Wind Shear")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential

    def _compute(self) -> xr.DataArray:
        return vertical_wind_shear(self._u_wind, self._v_wind, geopotential=self._geopotential)


class GradientRichardson(Diagnostic):
    _vws: xr.DataArray
    _brunt_vaisala: xr.DataArray

    def __init__(self, vws: xr.DataArray, brunt_vaisala: xr.DataArray) -> None:
        super().__init__("Richardson")
        self._vws = vws
        self._brunt_vaisala = brunt_vaisala

    def _compute(self) -> xr.DataArray:
        return self._brunt_vaisala / self._vws


class WindSpeed(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray) -> None:
        super().__init__("Wind Speed")
        self._u_wind = u_wind
        self._v_wind = v_wind

    def _compute(self) -> xr.DataArray:
        return np.hypot(self._u_wind, self._v_wind)  # pyright: ignore[reportReturnType]


class DeformationSquared(Diagnostic):
    _total_deformation: xr.DataArray

    def __init__(self, total_deformation: xr.DataArray) -> None:
        super().__init__("DEF Squared")
        self._total_deformation = total_deformation

    def _compute(self) -> xr.DataArray:
        return self._total_deformation * self._total_deformation


class WindDirection(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray) -> None:
        super().__init__("Wind Direction")
        self._u_wind = u_wind
        self._v_wind = v_wind

    def _compute(self) -> xr.DataArray:
        return wind_direction(self._u_wind, self._v_wind)


class HorizontalDivergence(Diagnostic):
    _divergence: xr.DataArray

    def __init__(self, divergence: xr.DataArray) -> None:
        super().__init__("Divergence")
        self._divergence = divergence

    def _compute(self) -> xr.DataArray:
        return np.abs(self._divergence)  # pyright: ignore[reportReturnType]


class MagnitudePotentialVorticity(Diagnostic):
    _potential_vorticity: xr.DataArray

    def __init__(self, potential_vorticity: xr.DataArray) -> None:
        super().__init__("|PV|")
        self._potential_vorticity = potential_vorticity

    def _compute(self) -> xr.DataArray:
        return np.abs(self._potential_vorticity)  # pyright: ignore[reportReturnType]


class GradientPotentialVorticity(Diagnostic):
    _potential_vorticity: xr.DataArray

    def __init__(self, potential_vorticity: xr.DataArray) -> None:
        super().__init__("|\\nabla PV|")
        self._potential_vorticity = potential_vorticity

    def _compute(self) -> xr.DataArray:
        return magnitude_of_geospatial_gradient(self._potential_vorticity)


class VerticalVorticitySquared(Diagnostic):
    _vorticity: xr.DataArray

    def __init__(self, vorticity: xr.DataArray) -> None:
        super().__init__("Vorticity Squared")
        self._vorticity = vorticity

    def _compute(self) -> xr.DataArray:
        return self._vorticity * self._vorticity


class DirectionalShear(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray) -> None:
        super().__init__("Directional Shear")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential

    def _compute(self) -> xr.DataArray:
        direction: xr.DataArray = wind_direction(self._u_wind, self._v_wind)
        z_axis: int = self._u_wind.get_axis_num("pressure_level")
        values_in_z_axis: np.ndarray = self._u_wind["pressure_level"].data
        if is_dask_collection(self._u_wind):
            directional_shear: xr.DataArray = xr.apply_ufunc(
                angles_gradient,
                direction,
                kwargs={"coord_values": values_in_z_axis, "target_axis": z_axis},
                dask="parallelized",
                output_dtypes=[np.float32],
            ).compute()
            return np.abs(altitude_derivative_on_pressure_level(directional_shear, self._geopotential))  # pyright: ignore[reportReturnType]
        return np.abs(altitude_derivative_on_pressure_level(direction, self._geopotential))  # pyright: ignore[reportReturnType]


class NestedGridModel1(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _total_deformation: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, total_deformation: xr.DataArray) -> None:
        super().__init__("NGM1")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._total_deformation = total_deformation

    def _compute(self) -> xr.DataArray:
        return wind_speed(self._u_wind, self._v_wind) * self._total_deformation


class NestedGridModel2(Diagnostic):
    _temperature: xr.DataArray
    _geopotential: xr.DataArray
    _total_deformation: xr.DataArray

    def __init__(self, temperature: xr.DataArray, geopotential: xr.DataArray, total_deformation: xr.DataArray) -> None:
        super().__init__("NGM2")
        self._temperature = temperature
        self._geopotential = geopotential
        self._total_deformation = total_deformation

    def _compute(self) -> xr.DataArray:
        vertical_temperature_gradient: xr.DataArray = np.abs(
            altitude_derivative_on_pressure_level(self._temperature, self._geopotential)
        )  # pyright: ignore[reportAssignmentType]
        return vertical_temperature_gradient * self._total_deformation


class BrownIndex1(Diagnostic):
    _vorticity: xr.DataArray
    _total_deformation: xr.DataArray

    def __init__(self, total_deformation: xr.DataArray, vorticity: xr.DataArray) -> None:
        super().__init__("Brown1")
        self._total_deformation = total_deformation.metpy.dequantify()
        self._vorticity = vorticity

    def _compute(self) -> xr.DataArray:
        abs_vorticity: xr.DataArray = absolute_vorticity(self._vorticity)
        return np.sqrt(0.3 * abs_vorticity + self._total_deformation)  # pyright: ignore[reportReturnType]


class BrownIndex2(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _brown_index1: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(
        self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray, brown_index_1: xr.DataArray
    ) -> None:
        super().__init__("Brown2")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential
        self._brown_index_1 = brown_index_1

    def _compute(self) -> xr.DataArray:
        vws: xr.DataArray = vertical_wind_shear(self._u_wind, self._v_wind, geopotential=self._geopotential)
        return (1 / 24) * self._brown_index_1 * vws * vws


class NegativeVorticityAdvection(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _vorticity: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, vorticity: xr.DataArray) -> None:
        super().__init__("NVA")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._vorticity = vorticity

    def _compute(self) -> xr.DataArray:
        abs_vorticity: xr.DataArray = absolute_vorticity(self._vorticity)
        x_component: xr.DataArray = (
            self._u_wind
            * spatial_gradient(abs_vorticity, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.X)["dfdx"]
        )
        y_component: xr.DataArray = (
            self._v_wind
            * spatial_gradient(abs_vorticity, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.Y)["dfdy"]
        )
        nva: xr.DataArray = -x_component - y_component
        return xr.where(nva < 0, 0, nva)


class DuttonIndex(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _geopotential: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray, geopotential: xr.DataArray) -> None:
        super().__init__("Dutton Index")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._geopotential = geopotential

    def horizontal_wind_shear(self, speed: xr.DataArray) -> xr.DataArray:
        x_component: xr.DataArray = (self._u_wind / speed) * spatial_gradient(
            speed, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.Y
        )["dfdy"]
        y_component: xr.DataArray = (self._v_wind / speed) * spatial_gradient(
            speed, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.X
        )["dfdx"]
        # Follows Sharman definition of horizontal wind shear
        # return x_component - y_component
        # Follows Dutton definition of horizontal wind shear
        return -x_component + y_component

    def _compute(self) -> xr.DataArray:
        speed: xr.DataArray = wind_speed(self._u_wind, self._v_wind)
        horizontal_wind_shear: xr.DataArray = self.horizontal_wind_shear(speed)
        vws: xr.DataArray = vertical_wind_shear(self._u_wind, self._v_wind, geopotential=self._geopotential)
        return 10.5 + 1.25 * horizontal_wind_shear + 0.25 * vws


class EDRLunnon(Diagnostic):
    _u_wind: xr.DataArray
    _v_wind: xr.DataArray
    _shear_deformation: xr.DataArray
    _stretching_deformation: xr.DataArray

    def __init__(
        self,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        shear_deformation: xr.DataArray,
        stretching_deformation: xr.DataArray,
    ) -> None:
        super().__init__("EDR Lunnon")
        self._u_wind = u_wind
        self._v_wind = v_wind
        self._shear_deformation = shear_deformation
        self._stretching_deformation = stretching_deformation

    def _compute(self) -> xr.DataArray:
        du_dp: xr.DataArray = self._u_wind.differentiate("pressure_level")
        dv_dp: xr.DataArray = self._v_wind.differentiate("pressure_level")
        return (
            (dv_dp * dv_dp) - (du_dp * du_dp)
        ) * self._stretching_deformation - 2 * du_dp * dv_dp * self._shear_deformation


class DiagnosticFactory:
    _data: "CATData"
    _richardson: xr.DataArray | None = None

    def __init__(self, data: "CATData") -> None:
        self._data = data

    @property
    def richardson(self) -> xr.DataArray:
        if self._richardson is None:
            self._richardson = self.create(TurbulenceDiagnostics.RICHARDSON).computed_value
        return self._richardson

    def create(self, diagnostic: TurbulenceDiagnostics) -> Diagnostic:
        match diagnostic:
            case TurbulenceDiagnostics.F2D:
                return Frontogenesis2D(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.potential_temperature(),
                    self._data.geopotential(),
                    self._data.velocity_derivatives(),
                )
            case TurbulenceDiagnostics.F3D:
                return Frontogenesis3D(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.potential_temperature(),
                    self._data.geopotential(),
                    self._data.divergence(),
                    self._data.velocity_derivatives(),
                )
            case TurbulenceDiagnostics.TEMPERATURE_GRADIENT:
                return HorizontalTemperatureGradient(self._data.temperature())
            case TurbulenceDiagnostics.ENDLICH:
                return Endlich(self._data.u_wind(), self._data.v_wind(), self._data.geopotential())
            case TurbulenceDiagnostics.TI1:
                return TurbulenceIndex1(
                    self._data.u_wind(), self._data.v_wind(), self._data.geopotential(), self._data.total_deformation()
                )
            case TurbulenceDiagnostics.TI2:
                return TurbulenceIndex2(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.velocity_derivatives(),
                    self._data.geopotential(),
                    self._data.total_deformation(),
                    self._data.divergence(),
                )
            case TurbulenceDiagnostics.NCSU1:
                return Ncsu1(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self.richardson,
                    self._data.velocity_derivatives(),
                    self._data.vorticity(),
                )
            case TurbulenceDiagnostics.COLSON_PANOFSKY:
                return ColsonPanofsky(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.altitude(),
                    self.richardson,
                    self._data.geopotential(),
                )
            case TurbulenceDiagnostics.UBF:
                return UBF(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.geopotential(),
                    self._data.vorticity(),
                    self._data.jacobian_horizontal_velocity(),
                )
            case TurbulenceDiagnostics.BRUNT_VAISALA:
                return BruntVaisalaFrequency(self._data.potential_temperature(), self._data.geopotential())
            case TurbulenceDiagnostics.VWS:
                return VerticalWindShear(self._data.u_wind(), self._data.v_wind(), self._data.geopotential())
            case TurbulenceDiagnostics.RICHARDSON:
                vws: Diagnostic = self.create(TurbulenceDiagnostics.VWS)
                brunt_vaisala: Diagnostic = self.create(TurbulenceDiagnostics.BRUNT_VAISALA)
                return GradientRichardson(vws.computed_value, brunt_vaisala.computed_value)
            case TurbulenceDiagnostics.WIND_SPEED:
                return WindSpeed(self._data.u_wind(), self._data.v_wind())
            case TurbulenceDiagnostics.DEF:
                return DeformationSquared(self._data.total_deformation())
            case TurbulenceDiagnostics.WIND_DIRECTION:
                return WindDirection(self._data.u_wind(), self._data.v_wind())
            case TurbulenceDiagnostics.HORIZONTAL_DIVERGENCE:
                return HorizontalDivergence(self._data.divergence())
            case TurbulenceDiagnostics.MAGNITUDE_PV:
                return MagnitudePotentialVorticity(self._data.potential_vorticity())
            case TurbulenceDiagnostics.PV_GRADIENT:
                return GradientPotentialVorticity(self._data.potential_vorticity())
            case TurbulenceDiagnostics.VORTICITY_SQUARED:
                return VerticalVorticitySquared(self._data.vorticity())
            case TurbulenceDiagnostics.DIRECTIONAL_SHEAR:
                return DirectionalShear(self._data.u_wind(), self._data.v_wind(), self._data.geopotential())
            case TurbulenceDiagnostics.NGM1:
                return NestedGridModel1(self._data.u_wind(), self._data.v_wind(), self._data.total_deformation())
            case TurbulenceDiagnostics.NGM2:
                return NestedGridModel2(
                    self._data.temperature(), self._data.geopotential(), self._data.total_deformation()
                )
            case TurbulenceDiagnostics.BROWN1:
                return BrownIndex1(self._data.vorticity(), self._data.total_deformation())
            case TurbulenceDiagnostics.BROWN2:
                brown1: Diagnostic = self.create(TurbulenceDiagnostics.BROWN1)
                return BrownIndex2(
                    self._data.u_wind(), self._data.v_wind(), self._data.geopotential(), brown1.computed_value
                )
            case TurbulenceDiagnostics.NVA:
                return NegativeVorticityAdvection(self._data.u_wind(), self._data.v_wind(), self._data.vorticity())
            case TurbulenceDiagnostics.DUTTON:
                return DuttonIndex(self._data.u_wind(), self._data.v_wind(), self._data.geopotential())
            case TurbulenceDiagnostics.EDR_LUNNON:
                return EDRLunnon(
                    self._data.u_wind(),
                    self._data.v_wind(),
                    self._data.shear_deformation(),
                    self._data.stretching_deformation(),
                )
            case _ as unreachable:
                assert_never(unreachable)


class DiagnosticSuite:
    _diagnostics: dict["DiagnosticName", Diagnostic]

    def __init__(self, factory: DiagnosticFactory, diagnostics: list[TurbulenceDiagnostics]) -> None:
        self._diagnostics: dict["DiagnosticName", Diagnostic] = {
            str(diagnostic): factory.create(diagnostic)
            for diagnostic in diagnostics  # TurbulenceDiagnostic
        }

    def computed_values(self, progress_description: str) -> Generator:
        for name, diagnostic in track(
            self._diagnostics.items(), description=progress_description
        ):  # DiagnosticName, Diagnostic
            yield name, diagnostic.computed_value


class CalibrationDiagnosticSuite(DiagnosticSuite):
    def __init__(self, factory: DiagnosticFactory, diagnostics: list[TurbulenceDiagnostics]) -> None:
        super().__init__(factory, diagnostics)

    def compute_thresholds(
        self, percentile_config: "TurbulenceThresholds"
    ) -> Mapping["DiagnosticName", "TurbulenceThresholds"]:
        return {
            name: TurbulenceIntensityThresholds(percentile_config, diagnostic).execute()
            for name, diagnostic in self.computed_values("Computing thresholds")  # DiagnosticName, xr.DataArray
        }

    def compute_distribution_parameters(self) -> Mapping["DiagnosticName", "HistogramData"]:
        return {
            name: DiagnosticHistogramDistribution(diagnostic).execute()
            for name, diagnostic in self.computed_values(
                "Computing distribution parameters"
            )  # DiagnosticName, xr.DataArray
        }


class EvaluationDiagnosticSuite(DiagnosticSuite):
    _probabilities: Mapping["DiagnosticName", xr.DataArray] | None = None
    _edr: Mapping["DiagnosticName", xr.DataArray] | None = None

    _severities: list["TurbulenceSeverity"] | None
    _pressure_levels: list[float] | None
    _probability_thresholds: Mapping["DiagnosticName", "TurbulenceThresholds"] | None
    _threshold_mode: TurbulenceThresholdMode | None
    _distribution_parameters: Mapping["DiagnosticName", "DistributionParameters"] | None

    def __init__(
        self,
        factory: DiagnosticFactory,
        diagnostics: list[TurbulenceDiagnostics],
        severities: list["TurbulenceSeverity"] | None = None,
        pressure_levels: list[float] | None = None,
        probability_thresholds: Mapping["DiagnosticName", "TurbulenceThresholds"] | None = None,
        threshold_mode: TurbulenceThresholdMode | None = None,
        distribution_parameters: Mapping["DiagnosticName", "DistributionParameters"] | None = None,
    ) -> None:
        super().__init__(factory, diagnostics)
        self._severities = severities
        self._pressure_levels = pressure_levels
        self._threshold_mode = threshold_mode
        for name in self._diagnostics:
            if distribution_parameters is not None and name not in distribution_parameters:
                raise KeyError(f"Diagnostic {name} has no distribution parameter")
            if probability_thresholds is not None and name not in probability_thresholds:
                raise KeyError(f"Diagnostic {name} has no probability threshold")
        self._probability_thresholds = probability_thresholds
        self._distribution_parameters = distribution_parameters

    @property
    def probabilities(self) -> Mapping["DiagnosticName", xr.DataArray]:
        if self._probabilities is None:
            if (
                self._severities is None
                or self._pressure_levels is None
                or self._probability_thresholds is None
                or self._threshold_mode is None
            ):
                raise ValueError("Probability of encountering turbulence of a given severity needs more inputs")
            self._probabilities = {
                name: TurbulenceProbabilityBySeverity(
                    diagnostic,
                    self._pressure_levels,
                    self._severities,
                    self._probability_thresholds[name],
                    self._threshold_mode,
                ).execute()
                for name, diagnostic in self.computed_values(
                    "Computing probability of encountering turbulence of a given severity"
                )
            }
            return self._probabilities

        return self._probabilities

    @property
    def edr(self) -> Mapping["DiagnosticName", xr.DataArray]:
        if self._edr is None:
            if self._distribution_parameters is None:
                raise ValueError("Computing EDR requires distribution parameters to be defined")

            edr = {}
            for name, diagnostic in self.computed_values("Computing EDR"):
                if name not in self._distribution_parameters:
                    raise ValueError(f"Distribution parameter for {name} is not defined")
                dist_params = self._distribution_parameters[name]
                edr[name] = TransformToEDR(diagnostic, dist_params.mean, dist_params.variance).execute()

            self._edr = edr
            return self._edr

        return self._edr
