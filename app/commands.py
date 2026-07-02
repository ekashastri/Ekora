"""
Command Pattern implementation for reliable Undo/Redo and action history.
"""
from typing import Protocol
from app.model import Document, Stroke, Shape, Layer
from typing import Union

class ICommand(Protocol):
    """Interface for all undoable commands."""
    def execute(self, document: Document) -> None:
        ...
        
    def undo(self, document: Document) -> None:
        ...

class AddItemCommand(ICommand):
    def __init__(self, item: Union[Stroke, Shape], layer_index: int):
        self.item = item
        self.layer_index = layer_index
        
    def execute(self, document: Document) -> None:
        document.layers[self.layer_index].items.append(self.item)
        
    def undo(self, document: Document) -> None:
        document.layers[self.layer_index].items.remove(self.item)

class RemoveItemCommand(ICommand):
    def __init__(self, item: Union[Stroke, Shape], layer_index: int):
        self.item = item
        self.layer_index = layer_index
        self._index_in_layer = -1
        
    def execute(self, document: Document) -> None:
        layer_items = document.layers[self.layer_index].items
        if self.item in layer_items:
            self._index_in_layer = layer_items.index(self.item)
            layer_items.remove(self.item)
            
    def undo(self, document: Document) -> None:
        if self._index_in_layer >= 0:
            document.layers[self.layer_index].items.insert(self._index_in_layer, self.item)

class ModifyShapeCommand(ICommand):
    def __init__(self, shape: Shape, old_params: dict, new_params: dict):
        self.shape = shape
        self.old_params = old_params.copy()
        self.new_params = new_params.copy()
        
    def execute(self, document: Document) -> None:
        self.shape.params = self.new_params.copy()
        
    def undo(self, document: Document) -> None:
        self.shape.params = self.old_params.copy()

class ModifyStrokeCommand(ICommand):
    def __init__(self, stroke: Stroke, old_points: list, new_points: list):
        self.stroke = stroke
        import copy
        self.old_points = copy.deepcopy(old_points)
        self.new_points = copy.deepcopy(new_points)
        
    def execute(self, document: Document) -> None:
        import copy
        self.stroke.points = copy.deepcopy(self.new_points)
        
    def undo(self, document: Document) -> None:
        import copy
        self.stroke.points = copy.deepcopy(self.old_points)

class ClearDocumentCommand(ICommand):
    def __init__(self):
        self.previous_layers = []
        
    def execute(self, document: Document) -> None:
        import copy
        self.previous_layers = copy.deepcopy(document.layers)
        document.layers = [Layer("Layer 1")]
        document.active_layer_index = 0
        
    def undo(self, document: Document) -> None:
        document.layers = self.previous_layers

class CommandManager:
    """Manages execution and undo/redo stacks for commands."""
    def __init__(self, document: Document):
        self.document = document
        self.undo_stack: list[ICommand] = []
        self.redo_stack: list[ICommand] = []
        
    def execute(self, command: ICommand) -> None:
        command.execute(self.document)
        self.undo_stack.append(command)
        self.redo_stack.clear()
        
    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        command = self.undo_stack.pop()
        command.undo(self.document)
        self.redo_stack.append(command)
        return True
        
    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        command = self.redo_stack.pop()
        command.execute(self.document)
        self.undo_stack.append(command)
        return True
        
    def clear_history(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
