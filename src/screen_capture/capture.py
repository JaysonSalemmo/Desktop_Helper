"""
Screen capture module.
Captures the screen and sends it to GPT-4o Vision for analysis.
Requires: openai API key in config.json
"""
import pyautogui
from pathlib import Path

SCREENSHOT_PATH = Path(__file__).parent.parent.parent / "data" / "screenshot.png"


def capture() -> Path:
    """Take a screenshot and save it. Returns the file path."""
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    screenshot = pyautogui.screenshot()
    screenshot.save(SCREENSHOT_PATH)
    return SCREENSHOT_PATH


def ask(question: str = "What do you see on this screen?") -> str:
    """Capture screen and ask GPT-4o Vision a question about it."""
    # TODO: wire up once OpenAI API key is configured
    # path = capture()
    # with open(path, "rb") as f:
    #     image_data = base64.b64encode(f.read()).decode()
    # client = OpenAI(api_key=...)
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": [
    #         {"type": "text", "text": question},
    #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
    #     ]}]
    # )
    # return response.choices[0].message.content
    raise NotImplementedError("Screen capture requires an OpenAI API key.")
