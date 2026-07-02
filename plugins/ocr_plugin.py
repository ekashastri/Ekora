"""
OCR Plugin Stub.
Demonstrates the architecture capability for Optical Character Recognition.
"""

from app.plugin_system import Plugin
from app.event_bus import EventBus, EventType
from app.services.providers import TesseractOCRProvider
from app.services.interfaces import IOCRProvider


class OCRPlugin(Plugin):
    @property
    def name(self) -> str:
        return "OCR Plugin"

    @property
    def description(self) -> str:
        return "Provides Optical Character Recognition capabilities."

    def __init__(self, provider: IOCRProvider):
        self.provider = provider
        
    def initialize(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.event_bus.subscribe(EventType.PLUGIN_OCR_REQUESTED, self.handle_ocr_request)

    def shutdown(self) -> None:
        self.event_bus.unsubscribe(EventType.PLUGIN_OCR_REQUESTED, self.handle_ocr_request)
        
    def handle_ocr_request(self, *args, **kwargs) -> None:
        image = kwargs.get("image", None)
        text = self.provider.extract_text(image)
        print(f"[OCR Plugin] OCR requested. Extracted: {text}")
        if text and not text.startswith("Pytesseract is not installed") and "Error" not in text:
            # Place the text roughly in the center or top-left
            params = {"text": text, "x": 100, "y": 100, "scale": 1.5}
            self.event_bus.publish(EventType.PLUGIN_ADD_SHAPE, shape_type="text", params=params, color=(255, 255, 255), size=2)
            self.event_bus.publish(EventType.FLASH_MESSAGE, "OCR process completed")
        else:
            self.event_bus.publish(EventType.FLASH_MESSAGE, f"OCR Failed: {text}")


def register_plugin() -> Plugin:
    return OCRPlugin(TesseractOCRProvider())
