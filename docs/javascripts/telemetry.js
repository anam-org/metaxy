// scarf.sh for telemetry collection ( it does not record personal data such as IP addresses)
(function () {
  var TOKEN = "22cb75dc-201e-4a72-9fb2-c3a53ce9207e";
  var PIXEL_BASE = "https://telemetry.metaxy.io/a.png?x-pxid=" + TOKEN + "=";
  var lastIdentifier = null;
  var pendingPixels = [];

  function currentIdentifier() {
    var path = window.location.pathname || "/";
    var trimmed = path.replace(/^\//, "");
    if (!trimmed) {
      trimmed = "index.html";
    }

    var query = window.location.search || "";
    var hash = window.location.hash || "";
    return trimmed + query + hash;
  }

  function sendPixel() {
    var identifier = currentIdentifier();
    if (identifier === lastIdentifier) {
      return;
    }
    lastIdentifier = identifier;

    var pixel = new Image(1, 1);
    pixel.referrerPolicy = "no-referrer";
    pixel.decoding = "async";
    pixel.src = PIXEL_BASE + encodeURIComponent(identifier);
    pendingPixels.push(pixel);

    var release = function () {
      var index = pendingPixels.indexOf(pixel);
      if (index !== -1) {
        pendingPixels.splice(index, 1);
      }
    };

    pixel.onload = release;
    pixel.onerror = release;
  }

  function track() {
    if (document.readyState === "loading") {
      document.addEventListener(
        "DOMContentLoaded",
        function handleReady() {
          document.removeEventListener("DOMContentLoaded", handleReady);
          sendPixel();
        },
      );
    } else {
      sendPixel();
    }

    if (window.document$ && typeof window.document$.subscribe === "function") {
      window.document$.subscribe(function () {
        sendPixel();
      });
    }
  }

  track();
})();
