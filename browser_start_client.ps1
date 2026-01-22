$chrome="C:\Users\27897\AppData\Local\Google\Chrome\Application\chrome.exe"
& $chrome `
  --remote-debugging-address=127.0.0.1 `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:LOCALAPPDATA\ChromeCDPProfile" `
  --profile-directory=Default
