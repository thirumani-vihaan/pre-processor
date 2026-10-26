"""Bounded real-browser smoke check; uses installed Edge and an isolated test server."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from lab.data import builtin_dataset


def main():
    screenshots = ROOT / "artifacts" / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dataprep-browser-") as temporary:
        server = make_server("127.0.0.1", 0, create_app(Path(temporary)), threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                context = browser.new_context(viewport={"width": 1500, "height": 1100}, device_scale_factor=1)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                def record_failure(request):
                    # Chromium hands download navigations off with ERR_ABORTED; completion is checked below.
                    if "/download/" in request.url and request.failure == "net::ERR_ABORTED":
                        return
                    errors.append(f"{request.url}: {request.failure}")

                page.on("requestfailed", record_failure)
                page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
                page.locator("#models input").first.wait_for()
                card_style = page.locator("#empty-state").evaluate("""el => ({
                    radius: parseFloat(getComputedStyle(el).borderTopLeftRadius),
                    border: parseFloat(getComputedStyle(el).borderTopWidth)
                })""")
                assert card_style["radius"] >= 10 and card_style["border"] == 1
                assert page.locator("h1").inner_text() == "Understand your data.\nImprove your models."
                page.screenshot(path=str(screenshots / "01-dashboard.png"), full_page=True)
                page.locator("#profile-button").click()
                page.locator("#profile-card:not([hidden])").wait_for()
                assert "12" in page.locator("#profile-content").inner_text()
                for checkbox in page.locator("#models input").all():
                    checkbox.set_checked(checkbox.get_attribute("value") in ("nb", "knn"))
                for checkbox in page.locator("#methods input").all():
                    checkbox.set_checked(checkbox.get_attribute("value") in ("iqr", "isolation", "lof"))
                page.locator("#run-button").click()
                page.locator("#results:not([hidden])").wait_for(timeout=180000)
                assert page.locator("#leaderboard tr").count() == 10
                assert page.locator(".winner-row").count() == 1
                page.screenshot(path=str(screenshots / "02-results.png"), full_page=True)
                page.get_by_role("button", name="02 Outliers", exact=True).click()
                page.locator("#outlier-view").select_option("isolation")
                assert page.locator("#scatter-chart svg circle").count() > 0
                assert page.locator("#scatter-chart svg path[data-anomaly='true']").count() > 0
                assert page.locator("#class-impact tbody tr").count() == 2
                page.screenshot(path=str(screenshots / "03-outliers.png"), full_page=True)
                page.get_by_role("button", name="03 Evaluation", exact=True).click()
                page.locator("#predict-button").click()
                page.locator(".prediction-card").wait_for()
                page.get_by_role("button", name="04 Reproduce", exact=True).click()
                assert page.locator("#export-links a").count() == 6
                with page.expect_download() as download:
                    page.locator("#bundle-link").click()
                assert download.value.suggested_filename.endswith(".zip")
                assert download.value.failure() is None
                page.locator("#dataset").select_option("upload")
                dataset = builtin_dataset("wine")
                frame = dataset.X.copy()
                frame["class"] = dataset.y
                page.locator("#csv-file").set_input_files({
                    "name": "wine.csv", "mimeType": "text/csv",
                    "buffer": frame.to_csv(index=False).encode(),
                })
                page.locator("#upload-status").filter(has_text="178 records").wait_for()
                assert page.locator("#target").input_value() == "class"
                page.locator("#profile-button").click()
                page.locator("#profile-content h3").filter(has_text="Uploaded CSV").wait_for()
                page.locator("#run-button").click()
                page.locator("#results:not([hidden])").wait_for(timeout=180000)
                assert page.locator("#result-title").inner_text() == "Uploaded CSV"
                expect(page.locator(".history-row")).to_have_count(2)
                page.locator(".history-row").nth(1).click()
                page.locator("#result-title").filter(has_text="Messy customer signals").wait_for()
                page.get_by_role("button", name="03 Evaluation", exact=True).click()
                page.locator("#prediction-json").fill("invalid json")
                page.locator("#predict-button").click()
                page.locator("#error:not([hidden])").wait_for()
                assert "valid JSON" in page.locator("#error").inner_text()
                assert page.locator("#prediction-output").inner_text() == ""
                page.locator(".history-row").first.click()
                page.locator("#result-title").filter(has_text="Uploaded CSV").wait_for()
                page.set_viewport_size({"width": 390, "height": 844})
                page.get_by_role("button", name="01 Overview", exact=True).click()
                page.screenshot(path=str(screenshots / "04-mobile.png"), full_page=True)
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                offenders = page.evaluate("""() => Array.from(document.querySelectorAll('body *'))
                    .filter(el => el.getBoundingClientRect().right > innerWidth + 2)
                    .slice(0, 12).map(el => `${el.tagName}.${el.className}: ${el.getBoundingClientRect().width}`)""")
                assert not overflow, f"Page has horizontal overflow on mobile: {offenders}"
                for width in (320, 768, 1024):
                    page.set_viewport_size({"width": width, "height": 900})
                    for tab in ("01 Overview", "02 Outliers", "03 Evaluation", "04 Reproduce"):
                        page.get_by_role("button", name=tab, exact=True).click()
                        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), f"Overflow at {width}px in {tab}"
                page.goto(f"http://127.0.0.1:{server.server_port}/guide", wait_until="networkidle")
                assert page.locator("h1").inner_text().startswith("Data Preprocessing")
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                assert not errors, errors
                print(json.dumps({"browser": "Microsoft Edge", "status": "passed", "console_errors": errors,
                                  "screenshots": str(screenshots), "checked_widths": [320, 390, 768, 1024, 1500],
                                  "mobile_horizontal_overflow": overflow}, indent=2))
                browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=10)


if __name__ == "__main__":
    main()
