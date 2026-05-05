# interface_minimal

Minimal browser UI for microphone streaming to Parakeet WebSocket endpoint.

## Quick start

1. Open terminal in this folder.
2. Start a local static server:

```bash
python -m http.server 8080
```

3. Open in browser:

- http://127.0.0.1:8080/index.html

4. Click **Start Mic** and allow microphone permission.
5. Click **Stop + Flush** to receive final transcript.

## Default endpoint

The page defaults to:

- ws://46.4.50.248:8001/ws/parakeet-stt

You can change the URL in the top input field.

## Notes

- Browser microphone access works reliably on secure contexts (HTTPS) or localhost.
- If transcript is empty, try speaking longer and then press **Stop + Flush**.
