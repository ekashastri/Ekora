"""
Core data model for the vector-based Document architecture.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union


@dataclass
class Stroke:
    """A freehand drawing stroke consisting of multiple points."""
    points: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    thickness: int
    is_eraser: bool = False


@dataclass
class Shape:
    """A recognized smart shape (e.g., circle, rectangle, arrow)."""
    shape_type: str
    params: Dict[str, Any]
    color: Tuple[int, int, int]
    thickness: int


@dataclass
class Layer:
    """A discrete layer containing strokes and shapes."""
    name: str
    items: List[Union[Stroke, Shape]] = field(default_factory=list)
    visible: bool = True
    locked: bool = False


@dataclass
class Document:
    """The root workspace document containing all layers and vector data."""
    layers: List[Layer] = field(default_factory=lambda: [Layer("Layer 1")])
    active_layer_index: int = 0

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_layer_index]
        
    def add_item(self, item: Union[Stroke, Shape]) -> None:
        self.active_layer.items.append(item)
        
    def remove_item(self, item: Union[Stroke, Shape]) -> None:
        for layer in self.layers:
            if item in layer.items:
                layer.items.remove(item)
                break

class Viewport:
    def __init__(self):
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.zoom: float = 1.0
        
    def screen_to_doc(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        return ((pt[0] - self.pan_x) / self.zoom, (pt[1] - self.pan_y) / self.zoom)
        
    def doc_to_screen(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        return (pt[0] * self.zoom + self.pan_x, pt[1] * self.zoom + self.pan_y)
