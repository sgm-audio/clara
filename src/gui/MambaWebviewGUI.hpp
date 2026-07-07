#pragma once

#include "clap/MambaClapPlugin.hpp"

#include <clap/ext/draft/webview.h>
#include <clap/ext/params.h>
#include <clap/host.h>
#include <clap/plugin.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace clara::mamba3 {

// ── Inline HTML page for the webview ────────────────────────────────────────
// Self-contained: all CSS and JS inline. No external resources.
static const char kWebviewHTML[] = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 16px;
    user-select: none; -webkit-user-select: none;
  }
  h1 { font-size: 14px; font-weight: 600; color: #00d4aa; margin-bottom: 16px; letter-spacing: 0.5px; }
  .param { margin-bottom: 14px; }
  .param label { display: block; font-size: 11px; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .param .value { float: right; font-size: 11px; color: #00d4aa; font-variant-numeric: tabular-nums; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 4px;
    background: #2a2a4a; border-radius: 2px; outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 14px; height: 14px;
    background: #00d4aa; border-radius: 50%; cursor: pointer;
    border: 2px solid #1a1a2e; box-shadow: 0 0 6px rgba(0,212,170,0.4);
  }
  input[type="range"]::-moz-range-thumb {
    width: 14px; height: 14px; background: #00d4aa; border-radius: 50%;
    cursor: pointer; border: 2px solid #1a1a2e;
  }
  .meter-container { margin-top: 16px; }
  .meter-label { font-size: 10px; color: #666; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
  .meter-bar { height: 8px; background: #2a2a4a; border-radius: 4px; overflow: hidden; margin-bottom: 6px; }
  .meter-fill { height: 100%; border-radius: 4px; transition: width 0.05s linear; }
  .meter-fill.in { background: linear-gradient(90deg, #00d4aa, #ffd700, #ff4444); }
  .meter-fill.out { background: linear-gradient(90deg, #00aaff, #00d4aa, #ffd700); }
  .info { margin-top: 16px; font-size: 10px; color: #555; text-align: center; }
  .bypass-btn {
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-size: 11px; cursor: pointer; text-transform: uppercase; letter-spacing: 0.5px;
    border: 1px solid #444; transition: all 0.15s;
  }
  .bypass-btn.on { background: #00d4aa; color: #1a1a2e; border-color: #00d4aa; }
  .bypass-btn.off { background: transparent; color: #888; }
</style>
</head>
<body>
  <h1>Mamba-3 SSM Processor</h1>

  <div class="param">
    <label>Bypass <span class="value" id="bypass-val">Off</span></label>
    <div id="bypass-btn" class="bypass-btn off" onclick="toggleBypass()">Off</div>
  </div>

  <div class="param">
    <label>Wet/Dry <span class="value" id="wetdry-val">100%</span></label>
    <input type="range" id="wetdry" min="0" max="100" value="100"
      oninput="paramChange('wetdry', this.value / 100)">
  </div>

  <div class="param">
    <label>Intensity <span class="value" id="intensity-val">50%</span></label>
    <input type="range" id="intensity" min="0" max="100" value="50"
      oninput="paramChange('intensity', this.value / 100)">
  </div>

  <div class="meter-container">
    <div class="meter-label">Input</div>
    <div class="meter-bar"><div class="meter-fill in" id="meter-in" style="width:0%"></div></div>
    <div class="meter-label">Output</div>
    <div class="meter-bar"><div class="meter-fill out" id="meter-out" style="width:0%"></div></div>
  </div>

  <div class="info" id="model-info">R=4 · 1.66M params · 6 layers</div>

  <script>
    function paramChange(id, value) {
      window.parent.postMessage({ type: 'param_change', id: id, value: value }, '*');
      var el = document.getElementById(id + '-val');
      if (el) el.textContent = Math.round(value * 100) + '%';
    }

    function toggleBypass() {
      var btn = document.getElementById('bypass-btn');
      var val = document.getElementById('bypass-val');
      var isOn = btn.classList.contains('on');
      btn.className = 'bypass-btn ' + (isOn ? 'off' : 'on');
      btn.textContent = isOn ? 'Off' : 'On';
      val.textContent = isOn ? 'Off' : 'On';
      window.parent.postMessage({ type: 'param_change', id: 'bypass', value: isOn ? 0 : 1 }, '*');
    }

    window.addEventListener('message', function(e) {
      var msg = e.data;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'param_update') {
        var el = document.getElementById(msg.id);
        if (el && el.tagName === 'INPUT') {
          el.value = msg.value * 100;
          var valEl = document.getElementById(msg.id + '-val');
          if (valEl) valEl.textContent = Math.round(msg.value * 100) + '%';
        }
      }
      if (msg.type === 'level') {
        var inEl = document.getElementById('meter-in');
        var outEl = document.getElementById('meter-out');
        if (inEl) inEl.style.width = Math.min(100, msg.input * 100) + '%';
        if (outEl) outEl.style.width = Math.min(100, msg.output * 100) + '%';
      }
      if (msg.type === 'model_info') {
        var infoEl = document.getElementById('model-info');
        if (infoEl && msg.text) infoEl.textContent = msg.text;
      }
    });
  </script>
</body>
</html>
)html";

// ── Webview extension state ────────────────────────────────────────────────
struct MambaWebviewState {
    const clap_host_webview_t *hostWebview{nullptr};
    bool webviewOpen{false};
};

// ── Webview extension implementation ───────────────────────────────────────
inline int32_t mamba_webview_get_uri(const clap_plugin_t *plugin, char *uri, uint32_t uri_capacity) {
    (void) plugin;
    // Return the HTML page as a data: URI
    const char *prefix = "data:text/html,";
    // URL-encode the HTML minimally: just encode <, >, ", &, space
    std::string encoded;
    for (const char *p = kWebviewHTML; *p; ++p) {
        switch (*p) {
            case '<':
                encoded += "%3C";
                break;
            case '>':
                encoded += "%3E";
                break;
            case '"':
                encoded += "%22";
                break;
            case '&':
                encoded += "%26";
                break;
            case ' ':
                encoded += "%20";
                break;
            case '#':
                encoded += "%23";
                break;
            case '%':
                encoded += "%25";
                break;
            default:
                encoded += *p;
        }
    }
    std::string full = prefix + encoded;
    int32_t len = static_cast<int32_t>(full.size()) + 1; // include null terminator
    if (uri && uri_capacity > 0) {
        std::strncpy(uri, full.c_str(), uri_capacity);
        uri[uri_capacity - 1] = '\0';
    }
    return len;
}

inline bool mamba_webview_get_resource(const clap_plugin_t *, const char *, char *, uint32_t, const clap_ostream_t *) {
    // All resources are inline in the HTML — no external files
    return false;
}

inline bool mamba_webview_receive(const clap_plugin_t *plugin, const void *buffer, uint32_t size) {
    auto *state = static_cast<MambaPluginState *>(plugin->plugin_data);
    if (!state || !buffer || size == 0) return false;

    // Parse JSON message from webview
    // Format: {"type":"param_change","id":"bypass","value":0.5}
    std::string msg(static_cast<const char *>(buffer), size);

    // Simple JSON parsing (no dependency — just what we need)
    auto findStr = [&](const std::string &key) -> std::string {
        auto pos = msg.find("\"" + key + "\"");
        if (pos == std::string::npos) return {};
        pos = msg.find(':', pos);
        if (pos == std::string::npos) return {};
        pos = msg.find_first_not_of(" \t\r\n", pos + 1);
        if (pos == std::string::npos) return {};
        if (msg[pos] == '"') {
            auto end = msg.find('"', pos + 1);
            if (end == std::string::npos) return {};
            return msg.substr(pos + 1, end - pos - 1);
        }
        // number
        auto end = msg.find_first_of(",}]", pos);
        if (end == std::string::npos) return msg.substr(pos);
        return msg.substr(pos, end - pos);
    };

    auto type = findStr("type");
    if (type == "param_change") {
        auto id = findStr("id");
        auto valStr = findStr("value");
        if (!id.empty() && !valStr.empty()) {
            double value = std::stod(valStr);
            clap_id paramId = 0;
            if (id == "bypass") paramId = 0;
            else if (id == "wetdry")
                paramId = 1;
            else if (id == "intensity")
                paramId = 2;
            else
                return false;

            // Store value in the plugin's atomic params (RT-safe)
            switch (paramId) {
                case 0:
                    state->bypass.store((float) value, std::memory_order_release);
                    break;
                case 1:
                    state->wetDry.store((float) value, std::memory_order_release);
                    break;
                case 2:
                    state->intensity.store((float) value, std::memory_order_release);
                    break;
            }
            // Notify host that params changed
            if (state->host && state->host->get_extension) {
                auto *hostParams =
                    static_cast<const clap_host_params_t *>(state->host->get_extension(state->host, CLAP_EXT_PARAMS));
                if (hostParams && hostParams->request_flush) hostParams->request_flush(state->host);
            }
        }
    }
    return true;
}

static const clap_plugin_webview_t mamba_webview_extension = {
    mamba_webview_get_uri,
    mamba_webview_get_resource,
    mamba_webview_receive,
};

} // namespace clara::mamba3
