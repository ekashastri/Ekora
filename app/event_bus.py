"""
Event Bus module - provides a lightweight publisher/subscriber pattern to decouple modules.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Any


class EventType(Enum):
    """Strongly typed events categorised by domain."""
    # Application
    APP_QUIT = auto()
    APP_STARTUP_COMPLETE = auto()
    
    # Workspace
    WORKSPACE_NEW = auto()
    WORKSPACE_OPEN = auto()
    WORKSPACE_SAVE = auto()
    WORKSPACE_SAVE_AS = auto()
    WORKSPACE_EXPORT = auto()
    WORKSPACE_SHOW_RECENT = auto()
    
    # Settings
    SETTINGS_OPEN = auto()
    SETTINGS_CHANGED = auto()
    
    # Canvas / Drawing
    CANVAS_CLEAR = auto()
    CANVAS_UNDO = auto()
    CANVAS_REDO = auto()
    BRUSH_SIZE_CHANGED = auto()
    COLOR_CHANGED = auto()
    SHAPE_RECOGNITION_TOGGLED = auto()
    
    # Tools / UI Info
    HELP_REQUESTED = auto()
    ABOUT_REQUESTED = auto()
    
    # Plugins
    PLUGIN_OCR_REQUESTED = auto()
    PLUGIN_DIAGRAM_REQUESTED = auto()
    PLUGIN_ADD_SHAPE = auto()
    FLASH_MESSAGE = auto()


class EventBus:
    """
    Lightweight Event Bus for internal module communication.
    Responsibilities:
    - subscribe()
    - unsubscribe()
    - publish()
    
    Does NOT store application state or contain business logic.
    """
    
    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[..., Any]) -> None:
        """Register a callback for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[..., Any]) -> None:
        """Unregister a callback from a specific event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def publish(self, event_type: EventType, *args: Any, **kwargs: Any) -> None:
        """Dispatch an event with optional arguments to all registered callbacks."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(*args, **kwargs)
