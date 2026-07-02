"""
Plugin system for modular feature expansion.
Supports dynamic loading of features like OCR and AI Diagram generation.
"""

from typing import Protocol, List
import importlib
import pkgutil
from pathlib import Path
import sys

from app.event_bus import EventBus


class Plugin(Protocol):
    @property
    def name(self) -> str:
        """Name of the plugin."""
        ...
        
    @property
    def description(self) -> str:
        """Description of what the plugin does."""
        ...
        
    def initialize(self, event_bus: EventBus) -> None:
        """Called when the plugin is loaded."""
        ...
        
    def shutdown(self) -> None:
        """Called when the application shuts down."""
        ...


class PluginManager:
    """Discovers and manages lifecycle of external plugins."""
    
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.plugins: List[Plugin] = []
        
    def discover_and_load(self, plugin_dir: str = "plugins") -> None:
        """Automatically load all plugins in the specified directory."""
        path = Path(__file__).parent.parent / plugin_dir
        
        # Create plugins directory if it doesn't exist
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)
            (path / "__init__.py").touch()
            
        # Ensure the parent directory is in sys.path
        parent_dir = str(path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # Dynamically load modules
        import plugins
        for _, module_name, _ in pkgutil.iter_modules(plugins.__path__, plugins.__name__ + "."):
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "register_plugin"):
                    plugin = module.register_plugin()
                    plugin.initialize(self.event_bus)
                    self.plugins.append(plugin)
                    print(f"Loaded plugin: {plugin.name} - {plugin.description}")
            except Exception as e:
                print(f"Failed to load plugin {module_name}: {e}")

    def shutdown_all(self) -> None:
        """Cleanly shutdown all plugins."""
        for plugin in self.plugins:
            try:
                plugin.shutdown()
            except Exception as e:
                print(f"Error shutting down plugin {plugin.name}: {e}")
