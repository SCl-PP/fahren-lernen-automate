# General Task

- We are independatly depth testing an iOS App running on mac os (as part of a third party security contract)
- For this, this (currently greenfield) directory is supposed to become a test harnish that can automatically navigate an app, by:
  - Capturing a screenshot
  - Processing it via gemini API, to determine what to press
  - Press the appropiate onscreen button
  - Capture another screenshot to see what happened
- In chat, by making decisions, walk me through what dependencies and libraries we need to make this work. The only constraints are
  - Will be an iPhone app running on my mac (this machine)
  - Must run fully automatic for an extended period of time, in the background
