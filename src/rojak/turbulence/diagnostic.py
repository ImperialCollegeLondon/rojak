from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from rojak.core.derivatives import GradientMode, VelocityDerivative, spatial_gradient
from rojak.turbulence.calculations import (
    GRAVITATIONAL_ACCELERATION,
    altitude_derivative_on_pressure_level,
    angles_gradient,
    magnitude_of_vector,
    vertical_wind_shear,
    wind_direction,
    wind_speed,
)

if TYPE_CHECKING:
    from rojak.core.derivatives import SpatialGradientKeys

type DiagnosticName = str


class Diagnostic(ABC):
    _name: DiagnosticName
    _computed_value: None | xr.DataArray = None

    def __init__(self, name: DiagnosticName) -> None:
        self._name = name

    @abstractmethod
    def _compute(self) -> xr.DataArray:
        pass

    # TODO: TEEST
    @property
    def name(self) -> DiagnosticName:
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

        d_direction_d_z: xr.DataArray = GRAVITATIONAL_ACCELERATION * (
            d_direction_d_p / self._geopotential.differentiate("pressure_level")
        )
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
