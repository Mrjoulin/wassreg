from .base import WassersteinRegressor, WassersteinCurveRegressor, WassersteinSurfaceRegressor
from .bezier_regressor import WassersteinBezierRegressor
from .bspline_regressor import WassersteinBSplineRegressor
from .bezier_surface_regressor import WassersteinBezierSurfaceRegressor
from .bspline_surface_regressor import WassersteinBSplineSurfaceRegressor

__all__ = [
    "WassersteinRegressor",
    "WassersteinCurveRegressor",
    "WassersteinSurfaceRegressor",
    "WassersteinBezierRegressor",
    "WassersteinBSplineRegressor",
    "WassersteinBezierSurfaceRegressor",
    "WassersteinBSplineSurfaceRegressor"
]
