from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from rojak.core.derivatives import GradientMode, VelocityDerivative, spatial_gradient
from rojak.turbulence.calculations import altitude_derivative_on_pressure_level

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
