from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to the app
            page.goto("http://localhost:8080")

            # Fill in the ticker and submit
            page.get_by_placeholder("Enter stock ticker...").fill("RELIANCE.NS")
            page.get_by_role("button", name="Get Signal").click()

            # Wait for the result to be displayed
            result_header = page.get_by_role("heading", name="Prediction for RELIANCE.NS")
            expect(result_header).to_be_visible()

            # Take a screenshot
            page.screenshot(path="jules-scratch/verification/verification.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run_verification()
