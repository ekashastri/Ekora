"""
AI Diagram Plugin Stub.
Demonstrates the architecture capability for AI-assisted diagram generation.
"""

from app.plugin_system import Plugin
from app.event_bus import EventBus, EventType
from app.services.providers import GeminiAIProvider
from app.services.interfaces import IAIProvider


class AIDiagramPlugin(Plugin):
    @property
    def name(self) -> str:
        return "AI Diagram Generator"

    @property
    def description(self) -> str:
        return "Converts rough sketches into structured AI-generated diagrams."

    def __init__(self, provider: IAIProvider):
        self.provider = provider
        
    def initialize(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.event_bus.subscribe(EventType.PLUGIN_DIAGRAM_REQUESTED, self.handle_diagram_request)

    def shutdown(self) -> None:
        self.event_bus.unsubscribe(EventType.PLUGIN_DIAGRAM_REQUESTED, self.handle_diagram_request)
        
    def handle_diagram_request(self, *args, **kwargs) -> None:
        image = kwargs.get("image", None)
        diagram = self.provider.generate_diagram("Generate from canvas", image)
        print(f"[AI Diagram Plugin] Diagram generated:\n{diagram}")
        if diagram and not diagram.startswith("google.generativeai is not installed") and "Error" not in diagram:
            params = {"text": diagram, "x": 100, "y": 200, "scale": 1.0}
            self.event_bus.publish(EventType.PLUGIN_ADD_SHAPE, shape_type="text", params=params, color=(0, 200, 200), size=2)
            self.event_bus.publish(EventType.FLASH_MESSAGE, "AI Diagram Generated")
        else:
            self.event_bus.publish(EventType.FLASH_MESSAGE, f"AI Diagram Failed: {diagram}")


def register_plugin() -> Plugin:
    return AIDiagramPlugin(GeminiAIProvider())
