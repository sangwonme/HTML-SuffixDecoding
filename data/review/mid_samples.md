# query_mid review samples (20 random from dev+test, seed 20261107)

Built by `scripts/build_query_mid.py`. Screenshot paths are relative to the GenerativeUI folder.
Checklist semantics: `type` must match the normalized category (type `Text` = keyword anywhere on the page); `keyword` = case-insensitive substring; `min_count` default 1.
Please mark each row: natural? (Y/N), checklist fair? (Y/N), comment.

## 1. `s2w_53519` (test, Sports)

- **query_mid**: Design a page displaying the multiple players in team in the San Diego Chargers app. Please put tabs including "Roster", a button mentioning name, a button that says "Pos", and a heading "Addae, Jahleel" on it.
- query_short (for comparison): Create a page displaying the multiple players in team.
- checklist (keyword, 4 items):
  - `{"type": "Multi-Tab", "keyword": "roster"}`
  - `{"type": "Text Button", "keyword": "name"}`
  - `{"type": "Text Button", "keyword": "pos"}`
  - `{"type": "Text", "keyword": "addae jahleel"}`
- screenshot: `data/processed/screenshots/s2w_53519.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 2. `s2w_16235` (test, Business)

- **query_mid**: Create a page showing activation code to continue on app. Include a top bar with a title about token, a button labeled "Cancel", and a "Continue" button.
- query_short (for comparison): Create a page showing activation code to continue on app.
- checklist (keyword, 3 items):
  - `{"type": "Toolbar", "keyword": "token"}`
  - `{"type": "Text Button", "keyword": "cancel"}`
  - `{"type": "Text Button", "keyword": "continue"}`
- screenshot: `data/processed/screenshots/s2w_16235.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 3. `s2w_55702` (test, Entertainment)

- **query_mid**: Make a page displaying multiple applications to share. Please put a line of text saying "Select email application." and a side drawer menu on it.
- query_short (for comparison): Create a page displaying multiple applications to share.
- checklist (mixed, 2 items):
  - `{"type": "Text", "keyword": "select email application"}`
  - `{"type": "Drawer", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_55702.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 4. `s2w_50494` (dev, Food & Drink)

- **query_mid**: For my Talabat app, I need a screen displaying the cuisine options to select. Please put a top bar titled "Choose Cuisine" and a heading "All Cuisines" on it.
- query_short (for comparison): Create a screen displaying the cuisine options to select.
- checklist (keyword, 2 items):
  - `{"type": "Toolbar", "keyword": "choose cuisine"}`
  - `{"type": "Text", "keyword": "all cuisines"}`
- screenshot: `data/processed/screenshots/s2w_50494.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 5. `s2w_27568` (test, Music & Audio)

- **query_mid**: Can you create a pop up alert to share via application? Add the text "Fake GPS - Search location" and a side drawer menu.
- query_short (for comparison): Create a pop up alert to share via application.
- checklist (mixed, 2 items):
  - `{"type": "Text", "keyword": "fake gps search location"}`
  - `{"type": "Drawer", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_27568.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 6. `s2w_53819` (test, Education)

- **query_mid**: Build a pop up alert to share via application. The screen needs a line of text saying "Share via", a navigation drawer and a list of items.
- query_short (for comparison): Create a pop up alert to share via application.
- checklist (mixed, 3 items):
  - `{"type": "Text", "keyword": "share via"}`
  - `{"type": "Drawer", "min_count": 1}`
  - `{"type": "List Item", "min_count": 2}`
- screenshot: `data/processed/screenshots/s2w_53819.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 7. `s2w_6666` (test, Auto & Vehicles)

- **query_mid**: Create an app screen showing image displaying a light house. It needs a "Cancel" button and a top toolbar.
- query_short (for comparison): Create a mobile app screen showing image displaying a light house.
- checklist (mixed, 2 items):
  - `{"type": "Text Button", "keyword": "cancel"}`
  - `{"type": "Toolbar", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_6666.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 8. `s2w_37173` (test, Food & Drink)

- **query_mid**: I need a screen displaying of search page in a recipes app. Please put a top bar titled "Search", a "meat" button and a button that says "fish & seafood" on it.
- query_short (for comparison): Create a display of search page in a recipes app.
- checklist (keyword, 3 items):
  - `{"type": "Toolbar", "keyword": "search"}`
  - `{"type": "Text Button", "keyword": "meat"}`
  - `{"type": "Text Button", "keyword": "fish seafood"}`
- screenshot: `data/processed/screenshots/s2w_37173.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 9. `s2w_56049` (dev, Social)

- **query_mid**: Can you create a page displaying list of social media applications? It should have a check box for "Facebook" and a checkbox for "twitter".
- query_short (for comparison): Create a page displaying list of social media applications.
- checklist (keyword, 2 items):
  - `{"type": "Checkbox", "keyword": "facebook"}`
  - `{"type": "Checkbox", "keyword": "twitter"}`
- screenshot: `data/processed/screenshots/s2w_56049.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 10. `s2w_930` (test, Auto & Vehicles)

- **query_mid**: Design a page for registration of car for checking stolen car details in the Stolen Car Checker app. Include a line of text saying "Car Registration", a button labeled "Register", and text input fields.
- query_short (for comparison): Create a page for registration of car for checking stolen car details.
- checklist (mixed, 3 items):
  - `{"type": "Text", "keyword": "car registration"}`
  - `{"type": "Text Button", "keyword": "register"}`
  - `{"type": "Input", "min_count": 2}`
- screenshot: `data/processed/screenshots/s2w_930.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 11. `s2w_54005` (test, Education)

- **query_mid**: Design a page displaying to add notes in learning app. Make sure it has a header bar that says "Notes" and an image.
- query_short (for comparison): Create a page displaying to add notes in learning app.
- checklist (mixed, 2 items):
  - `{"type": "Toolbar", "keyword": "notes"}`
  - `{"type": "Image", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_54005.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 12. `s2w_60897` (dev, Books & Reference)

- **query_mid**: I need a welcome page of a gaming app. It should have an app bar with the title "Choose a path", the text "New to French?", and an image.
- query_short (for comparison): Create a welcome page of a gaming app.
- checklist (mixed, 3 items):
  - `{"type": "Toolbar", "keyword": "choose a path"}`
  - `{"type": "Text", "keyword": "new to french"}`
  - `{"type": "Image", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_60897.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 13. `s2w_60858` (test, Entertainment)

- **query_mid**: Design a screen displaying of terms and conditions in the Now Player app. It should have a heading "Terms & Conditions", a "Cancel" button and a button labeled "Accept".
- query_short (for comparison): Create a pop up displaying terms and conditions.
- checklist (keyword, 3 items):
  - `{"type": "Text", "keyword": "terms conditions"}`
  - `{"type": "Text Button", "keyword": "cancel"}`
  - `{"type": "Text Button", "keyword": "accept"}`
- screenshot: `data/processed/screenshots/s2w_60858.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 14. `s2w_70336` (test, Food & Drink)

- **query_mid**: Make a home page displaying list of settings for cooking app. Include a header bar that says "Settings" and a heading "Change language".
- query_short (for comparison): Create a home page displaying list of settings for cooking app.
- checklist (keyword, 2 items):
  - `{"type": "Toolbar", "keyword": "settings"}`
  - `{"type": "Text", "keyword": "change language"}`
- screenshot: `data/processed/screenshots/s2w_70336.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 15. `s2w_39310` (test, Communication)

- **query_mid**: Design a page displaying information about calling application. The screen needs a heading "Unknown" and a picture.
- query_short (for comparison): Create a page displaying information about calling application.
- checklist (mixed, 2 items):
  - `{"type": "Text", "keyword": "unknown"}`
  - `{"type": "Image", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_39310.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 16. `s2w_1853` (dev, Health & Fitness)

- **query_mid**: Create a page of an eye app asking to start a test. Please put a button labeled "how-to" and a button that says "start" on it.
- query_short (for comparison): Create a page of an eye app asking to start a test.
- checklist (keyword, 2 items):
  - `{"type": "Text Button", "keyword": "how to"}`
  - `{"type": "Text Button", "keyword": "start"}`
- screenshot: `data/processed/screenshots/s2w_1853.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 17. `s2w_53817` (test, Education)

- **query_mid**: Create a page showing information of time available for the TED app. Please put a note about much, a button that says "Continue", a button labeled "I listen to color" and a top toolbar on it.
- query_short (for comparison): Create a page showing information of time available.
- checklist (mixed, 4 items):
  - `{"type": "Text", "keyword": "much"}`
  - `{"type": "Text Button", "keyword": "continue"}`
  - `{"type": "Text Button", "keyword": "i listen to color"}`
  - `{"type": "Toolbar", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_53817.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 18. `s2w_62629` (test, Lifestyle)

- **query_mid**: Can you create a page showing recipe cooking details of a cooking app? Please put a header bar that says "omlette", a line of text saying "Show: Melting Pot", a button that says "Recipe", and a "Notes" button on it.
- query_short (for comparison): Create a page showing recipe cooking details of a cooking app.
- checklist (keyword, 4 items):
  - `{"type": "Toolbar", "keyword": "omlette"}`
  - `{"type": "Text", "keyword": "show melting pot"}`
  - `{"type": "Text Button", "keyword": "recipe"}`
  - `{"type": "Text Button", "keyword": "notes"}`
- screenshot: `data/processed/screenshots/s2w_62629.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 19. `s2w_56635` (test, News & Magazines)

- **query_mid**: Make an app screen showing latest updates of news on a news app. It needs a line of text saying "WMUR On Demand", a button that says "By Kim Bergeron", and an image.
- query_short (for comparison): Create a mobile app screen showing latest updates of news on a news app.
- checklist (mixed, 3 items):
  - `{"type": "Text", "keyword": "wmur on demand"}`
  - `{"type": "Text Button", "keyword": "by kim bergeron"}`
  - `{"type": "Image", "min_count": 1}`
- screenshot: `data/processed/screenshots/s2w_56635.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:

## 20. `s2w_15124` (test, Shopping)

- **query_mid**: I need a page showing about different countries option. It should have a heading "Choose Your Country" and a scrollable list.
- query_short (for comparison): Create a page showing about different countries option.
- checklist (mixed, 2 items):
  - `{"type": "Text", "keyword": "choose your country"}`
  - `{"type": "List Item", "min_count": 2}`
- screenshot: `data/processed/screenshots/s2w_15124.jpg`
- leak_ok: True
- review: natural? __  fair? __  comment:
