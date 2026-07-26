"""Question 5: five-UAV, three-missile route optimization."""

from .model import Route, decode_route_particle, solve_integer_routes

__all__ = ["Route", "decode_route_particle", "solve_integer_routes"]
